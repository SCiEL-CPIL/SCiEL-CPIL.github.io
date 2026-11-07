import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Upper Confidence Bound for minimization
    # We want low mu and high sigma (uncertainty)
    # Score should be high for good candidates
    # Good candidate: low mu, high sigma
    # Score = -mu + beta * sigma
    beta = 2.0
    ucb = -mu + beta * sigma_safe
    
    # Cost-aware score
    score = ucb / cost
    
    # Handle any non-finite values
    score = np.where(np.isfinite(score), score, -1e9)
    
    return score
