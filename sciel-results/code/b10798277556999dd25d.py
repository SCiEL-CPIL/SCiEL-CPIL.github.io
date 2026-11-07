"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a variant of Expected Improvement (EI) adjusted for cost,
    with a floor on sigma to avoid division issues and ensure stability.
    
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
    
    # Standardize the improvement
    z = (f_best - mu) / sigma
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (clip small negative values due to numerical noise)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost, but add a small epsilon to cost to avoid division by zero
    cost_safe = np.maximum(cost, 1e-6)
    
    # Return EI per unit cost
    scores = ei / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
