import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Use UCB (Upper Confidence Bound) for minimization
    # Score = -mu + beta * sigma
    # Higher score is better.
    # Low mu (exploitation) and high sigma (exploration) increase score.
    
    beta = 1.0
    
    # Calculate UCB score
    scores = -mu + beta * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
