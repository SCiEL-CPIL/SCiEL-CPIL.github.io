"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware UCB acquisition function for Bayesian optimization.
    Uses Upper Confidence Bound (UCB) adjusted for cost.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP at each candidate point.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP at each candidate point.
    f_best : float
        Best observed value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more promising).
    """
    # Ensure sigma is positive and has a small floor to avoid numerical issues
    sigma = np.maximum(sigma, 1e-6)
    
    # UCB parameter (exploration-exploitation trade-off)
    beta = 1.0
    
    # For minimization, we want to maximize -mu + beta * sigma
    # This is equivalent to minimizing mu - beta * sigma
    ucb = -mu + beta * sigma
    
    # Cost-aware: divide by cost
    cost_safe = np.maximum(cost, 1e-6)
    
    scores = ucb / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
