"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function for minimization.
    
    The score is defined as -mu + beta * sigma.
    We maximize this score to select the next point.
    -mu favors points with low expected function values.
    beta * sigma favors points with high uncertainty.
    """
    beta = 1.5
    
    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 0.0)
    
    # Calculate scores
    scores = -mu + beta * sigma_safe
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
