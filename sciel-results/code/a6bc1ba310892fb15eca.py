"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Thompson Sampling proxy: maximize (f_best - mu) / (sigma + epsilon)
    # This encourages points with high potential improvement relative to uncertainty.
    # Higher score means more likely to be selected.
    
    epsilon = 1e-8
    sigma_safe = np.maximum(sigma, epsilon)
    
    # Normalized improvement heuristic
    scores = (f_best - mu) / sigma_safe
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e9, neginf=-1e9)
    
    return scores
# EVOLVE-BLOCK-END
