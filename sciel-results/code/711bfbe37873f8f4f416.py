import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB) for minimization
    # We want to minimize f(x).
    # The lower bound of the prediction is mu - beta * sigma.
    # We want to select the point with the lowest lower bound.
    # To make this a "higher is better" score, we negate it:
    # Score = -(mu - beta * sigma) = -mu + beta * sigma
    
    # Choose beta. 2.0 is a standard choice for BO.
    beta = 2.0
    
    # Ensure numerical stability
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate LCB score
    scores = -mu + beta * sigma_safe
    
    return scores
