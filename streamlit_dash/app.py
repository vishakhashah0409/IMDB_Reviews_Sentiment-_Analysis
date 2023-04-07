import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from tkinter import *
from sklearn.metrics import confusion_matrix



st.set_page_config(
    page_title="IMDB sentiment analysis",
    page_icon="🕸",
    layout="wide",
    initial_sidebar_state="expanded")

# Load the data
data1 = pd.read_csv("C:/Users/visha/OneDrive/Documents/processeddata.csv")
tfidf = TfidfVectorizer()

with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Load the saved model
with open('support_vector_machine_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the Streamlit app
def app():
    # Set the app title
    st.title('IMDb Movie Review Sentiment Analysis')
    st.write("This dashboard performs sentiment analysis on IMDb movie reviews. The dataset contains 50,000 reviews labeled as either positive or negative.")
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Add a sidebar with options
    option = st.sidebar.selectbox('Select an option:', ('Home', 'show Dataset', 'Model Performance'))
    # st.sidebar.markdown("##Upload reviews data :")
    uploaded_folder = st.sidebar.file_uploader("Choose a csv file")
    # Load the IMDb review dataset
    if uploaded_folder is not None:
        data1 = pd.read_csv(uploaded_folder)
    else:
        data1 = pd.read_csv("C:/Users/visha/OneDrive/Documents/processeddata.csv")
    #sia = SentimentIntensityAnalyzer()
    #data1["Sentiment"] = data1["processed"].apply(lambda x: sia.polarity_scores(x)["compound"])

    from textblob import TextBlob
    # Function to analyze sentiment
    def sentiment_analysis(text):
        from textblob import TextBlob
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0:
            return 'Positive'
        else:
            return 'Negative'
        

    # If a file is uploaded, display sentiment analysis results
    uploaded_file = st.sidebar.file_uploader("Choose a file")
    if uploaded_file is not None:
        data1 = pd.read_csv(uploaded_file)
        data1['Sentiment'] = data1['processed'].apply(sentiment_analysis)
        st.write(data1)


    
    # Home page
    if option == 'Home':
        st.write('Welcome to the IMDb movie review sentiment analysis dashboard!')
        st.write('Please select a task from the sidebar.')
        st.set_option('deprecation.showPyplotGlobalUse', False)

        # Sentiment Analyzer
        st.subheader('Sentiment Analyzer')
        processed = st.text_input('Enter a movie review:', '')
        
        # Make a prediction based on the user input
        if st.button('Predict'):
            review_tfidf = tfidf.transform([processed])
            prediction = model.predict(review_tfidf)[0]
            if prediction == 1:
                st.write('Prediction:', prediction,'😃')
            else:
                st.write('Prediction:', prediction,'😞')

        # Show most frequent words  
        st.subheader('Most Frequent Words')
        feature_names = tfidf.get_feature_names_out()
        coef = model.coef_.ravel()
        top_positive_words = pd.Series(coef, index=feature_names).nlargest(10)
        top_negative_words = pd.Series(coef, index=feature_names).nsmallest(10)
        top_words = pd.concat([top_positive_words, top_negative_words])
        plt.figure(figsize=(15, 4))
        sns.barplot(x=top_words.index, y=top_words.values)
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Words')
        plt.ylabel('Weight')
        plt.title('Top Words Contributing to Sentiment')
        st.pyplot()

        # Create plot of review length distribution
        data1['review length(word count)'] = data1['processed'].apply(lambda x: len(x.split()))
        fig1 = px.histogram(data1, x='review length(word count)', nbins=50, title='Review Length Distribution')
        st.plotly_chart(fig1)
        
        # Show sentiment distribution
        st.subheader('Sentiment Distribution')
        sentiment_counts = data1['sentiment'].value_counts()
        sns.histplot(data=data1, x='sentiment')
        if not sentiment_counts.empty:
            plt.xticks(np.arange(2), ['Negative', 'Positive'])
            plt.xlabel('Sentiment')
            plt.ylabel('Count')
            plt.title('Sentiment Distribution')
            st.pyplot()  
        
        # Define functions for generating wordclouds
        def plot_wordcloud(data1, sentiment):
            mask=None
            if sentiment == 'Positive':
                    subset = data1[data1['sentiment'] == 1]
                    title = "Positive Wordcloud"
                    colormap = "gist_rainbow_r"
            else:
                    subset = data1[data1['sentiment'] == 0]
                    title = "Negative Wordcloud"
                    colormap = "CMRmap"
            text = " ".join(subset["processed"])
            wc = WordCloud(background_color="white", mask=mask, colormap=colormap, max_words=200, contour_width=3, contour_color='firebrick').generate(text)
            plt.figure(figsize=(8,8))
            plt.imshow(wc, interpolation='bilinear')
            plt.title(title, fontsize=12)
            plt.axis("off")
            plt.tight_layout(pad=0)
            st.pyplot()
        # Display the wordclouds in the main area
        st.markdown("## Positive and Negative Review Wordclouds")
        if "processed" not in data1.columns:
            st.warning("Please upload a valid IMDb review dataset")
        else:
            plot_wordcloud(data1, "Positive")
            plot_wordcloud(data1, "Negative")
  
    # View dataset page
    elif option == 'show Dataset':
        st.write('This is the IMDb movie review dataset used in this dashboard.')
        # Drop the "column_name" column from the DataFrame
        #data1 = data1.drop(columns=["Unnamed: 0"])
        #data1 = data1.drop(columns=["length"])
        st.write(data1)
    # Train model page
    elif option == 'Model Performance':
        st.write('This may take a minute or two.')
        # Load the test data
        test_data = pd.read_csv('X_test.csv')
        test_tfidf = tfidf.transform(test_data['processed'])
        svm_tfidf_preds = model.predict(test_tfidf)
        # Preprocess the test data
        X_test = test_data['processed']
        Y_test = test_data['sentiment']
        # Generate the confusion matrix
        tfidf_matrix = confusion_matrix(Y_test, svm_tfidf_preds)
        # Define the labels for the confusion matrix
        labels = ['Positive', 'Negative']
        
        # Calculate the accuracy of the SVM model
        accuracy = np.mean(svm_tfidf_preds == Y_test)
        # Display the accuracy
        st.write('Accuracy:', accuracy)
        # Create a heatmap of the confusion matrix
        fig, ax = plt.subplots()
        sns.heatmap(tfidf_matrix/np.sum(tfidf_matrix), annot=True, cmap='Blues', fmt='.2%', xticklabels=labels, yticklabels=labels)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
                #return prediction
            #return review
       #return option
    #return option
# Run the Streamlit app

if __name__ == "__main__":
    app()



