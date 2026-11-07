import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Use UCB (Lower Confidence Bound for minimization)
    # Score = (-mu + beta * sigma) / cost
    # Higher score is better.
    
    beta = 2.5
    
    # Ensure sigma is non-negative and handle potential numerical issues
    sigma = np.maximum(sigma, 1e-6)
    
    # Calculate LCB for minimization: mu - beta * sigma
    # We want to maximize -LCB = -mu + beta * sigma
    score = (-mu + beta * sigma) / cost
    
    return score
