"""Rewritten program with UCB acquisition."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB) for minimization.
    Returns scores where higher is better (i.e., lower predicted value + uncertainty bonus).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # UCB for minimization:
    # Lower Confidence Bound (LCB) = mu - beta * sigma
    # We want to minimize LCB.
    # Score = -LCB = -mu + beta * sigma
    # Higher score means lower LCB (more promising).
    
    beta = 4.0
    scores = -mu + beta * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
