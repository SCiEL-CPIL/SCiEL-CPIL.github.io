import numpy as np

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization
    # Score = f_best - mu + beta * sigma
    # Higher score indicates a better candidate for the next evaluation.
    
    # Beta controls the trade-off between exploration and exploitation.
    # A value around 2.0 is a common starting point for standardized GPs on unit domains.
    beta = 2.0
    
    # Ensure sigma is non-negative to avoid unexpected behavior if numerical errors occur
    sigma_safe = np.maximum(sigma, 0.0)
    
    # Calculate UCB score
    score = f_best - mu + beta * sigma_safe
    
    return score
