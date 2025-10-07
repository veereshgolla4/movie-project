# Movie Rating Prediction using Regression

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load dataset
# Example dataset (You can replace with a larger real dataset like IMDB/RottenTomatoes)
data = {
    'Genre': ['Action', 'Comedy', 'Drama', 'Action', 'Comedy', 'Drama', 'Action', 'Drama', 'Comedy', 'Action'],
    'Director': ['Nolan', 'Apatow', 'Spielberg', 'Nolan', 'Apatow', 'Spielberg', 'Nolan', 'Spielberg', 'Apatow', 'Nolan'],
    'Actors': ['Bale', 'Rogen', 'Hanks', 'Hardy', 'Hill', 'DiCaprio', 'Bale', 'Tom Hanks', 'Jonah Hill', 'Hardy'],
    'Rating': [8.8, 7.2, 8.5, 8.4, 6.9, 8.7, 8.9, 8.6, 7.0, 8.3]
}
df = pd.DataFrame(data)

print("Sample Movie Dataset:\n", df)

# Step 3: Preprocessing
# Convert categorical columns into numerical features
encoder = LabelEncoder()
df['Genre'] = encoder.fit_transform(df['Genre'])
df['Director'] = encoder.fit_transform(df['Director'])
df['Actors'] = encoder.fit_transform(df['Actors'])

print("\nEncoded Dataset:\n", df)

# Step 4: Split features and target
X = df[['Genre', 'Director', 'Actors']]
y = df['Rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Predictions
y_pred = model.predict(X_test)

# Step 7: Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("Mean Squared Error:", mse)
print("R² Score:", r2)

# Step 8: Compare actual vs predicted
results = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
print("\nComparison of Actual vs Predicted:\n", results)

# Step 9: Visualization
plt.figure(figsize=(6,4))
sns.scatterplot(x=y_test, y=y_pred, s=100, color="blue")
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Actual vs Predicted Movie Ratings")
plt.show()
