import numpy as np

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization
    # Score = -mu + kappa * sigma
    # Higher score means more promising (lower expected value or higher uncertainty)
    kappa = 1.0
    scores = -mu + kappa * sigma
    
    # Safety: ensure finite output
    # If any value is not finite, replace with a very low score
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
