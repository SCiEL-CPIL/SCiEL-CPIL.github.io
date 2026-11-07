"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Upper Confidence Bound for minimization
    # We want to maximize the score.
    # Score = -mu + beta * sigma
    # This balances finding low means (exploitation) with high uncertainty (exploration).
    beta = 1.0
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
