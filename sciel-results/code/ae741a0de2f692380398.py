"""Acquisition function for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Combined acquisition: Expected Improvement + UCB exploration bonus.
    
    Parameters:
    -----------
    mu : ndarray, shape (n,)
        Mean of GP posterior at candidate points
    sigma : ndarray, shape (n,)
        Standard deviation of GP posterior at candidate points
    f_best : float
        Best (lowest) observed objective value so far
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores (higher = more promising)
    """
    # Add small epsilon to avoid division by zero
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB exploration bonus: beta * sigma
    # Tune beta to balance exploitation and exploration
    beta = 0.5
    
    # Combined score
    scores = ei + beta * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
