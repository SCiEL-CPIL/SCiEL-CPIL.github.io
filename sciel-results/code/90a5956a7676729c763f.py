import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
       # Ensure sigma is positive to avoid division by zero
       sigma = np.maximum(sigma, 1e-8)
       
       # Calculate z-score
       z = (f_best - mu) / sigma
       
       # Expected Improvement (EI)
       # For minimization, EI is the expected amount by which f(x) improves f_best
       ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
       
       # Probability of Improvement (POI)
       poi = norm.cdf(z)
       
       # Exploration bonus: Weighted by POI to focus on plausible improvements
       # This adds a smooth exploration signal that doesn't explode in bad regions
       exploration = sigma * poi
       
       # Combine EI and Exploration
       # Using a weight of 0.5 balances exploitation and exploration
       score = ei + 0.5 * exploration
       
       # Ensure scores are finite
       score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
       
       return score
