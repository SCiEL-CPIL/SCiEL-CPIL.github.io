"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Lower Confidence Bound (LCB) acquisition function for minimization.
    LCB encourages exploration by subtracting a confidence term from the mean.
    For minimization, we minimize mu - beta * sigma.
    The score returned is -LCB = -mu + beta * sigma, so higher scores are better.
    """
    # Beta controls the trade-off between exploration and exploitation.
    # A value > 1 encourages exploration.
    beta = 1.5
    
    # Handle potential numerical issues with sigma
    eps = 1e-12
    safe_sigma = np.maximum(sigma, eps)
    
    # LCB = mu - beta * sigma
    # We want to maximize the score, so we return -LCB
    score = -mu + beta * safe_sigma
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
