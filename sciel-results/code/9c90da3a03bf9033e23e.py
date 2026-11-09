"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function.
    
    Args:
        mu: array of shape (n,) - posterior means
        sigma: array of shape (n,) - posterior standard deviations
        f_best: float - best observed value so far
    
    Returns:
        array of shape (n,) - acquisition scores (higher is better for selection)
    """
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-12)
    
    # UCB for minimization: we want to maximize -mu + beta * sigma
    # beta controls the exploration-exploitation trade-off
    beta = 1.5
    
    ucb = -mu + beta * sigma
    
    # Ensure all scores are finite
    ucb = np.where(np.isfinite(ucb), ucb, 0.0)
    
    return ucb
# EVOLVE-BLOCK-END
