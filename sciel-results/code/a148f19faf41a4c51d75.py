"""Acquisition function using Upper Confidence Bound (UCB)."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB) for minimization.
    
    UCB(x) = -mu(x) + beta * sigma(x)
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : array of shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        The lowest observed value so far.
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 0.0)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 0.0)
    
    # Ensure mu is finite
    mu_safe = np.where(np.isfinite(mu), mu, np.inf)
    
    # Beta parameter for exploration vs exploitation trade-off
    # A value of 1.0 is a standard starting point.
    # Higher values encourage more exploration.
    beta = 1.0
    
    # UCB for minimization: we want to minimize f(x).
    # The acquisition score is designed to be high for points with low expected value
    # and high uncertainty.
    scores = -mu_safe + beta * sigma_safe
    
    # Handle infinities
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
