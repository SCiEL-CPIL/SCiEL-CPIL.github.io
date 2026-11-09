import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Lower Confidence Bound (LCB) for minimization.
    # We want to find the point that minimizes the function.
    # LCB(x) = mu(x) - beta * sigma(x)
    # A lower LCB is better.
    # To convert to a "higher is better" score, we use -LCB(x).
    # We also divide by cost to penalize expensive evaluations.
    
    # Exploration parameter
    beta = 2.0
    
    # Ensure sigma is positive to avoid division by zero or negative variance issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate the score
    # Score = (beta * sigma - mu) / cost
    score = (beta * sigma_safe - mu) / cost
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
