#The following are the libaries that are required to perform the task
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the csvb data file  into a pandas dataframe
adv_data = pd.read_csv("C:/Users/nares/Downloads/Advertising.csv")

# Split the data into training and testing sets
training_data = adv_data.iloc[:80]
testing_data = adv_data.iloc[80:]

#training dataset
training_data

#testing data set
testing_data

X_test
y_test

model = LinearRegression()
model.fit(X_train, y_train)

model

# Make predictions on the test data
y_pred = model.predict(X_test)
y_pred

# Evaluate the model performance
from sklearn.metrics import r2_score
r2_score = r2_score(y_test, y_pred)
print("R-squared score: ", r2_score)
