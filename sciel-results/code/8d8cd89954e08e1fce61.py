"""Cost-aware acquisition function for Bayesian optimization (three hump camel 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


# Module-level deadline tracking
_DEADLINE = None
_DEADLINE_LIMIT = 290.0  # 300s limit, keep 10s safety margin


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions for 2000 candidate points.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions for 2000 candidate points.
    f_best : float
        Best observed function value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    global _DEADLINE, _DEADLINE_LIMIT
    
    # Initialize deadline if not already set
    if _DEADLINE is None:
        _DEADLINE = time.time() + _DEADLINE_LIMIT
    
    # Check if we're close to the deadline
    if time.time() > _DEADLINE - 5.0:
        # Return a safe fallback: just use normalized negative mu
        # This is fast and won't cause issues
        return -mu / (np.std(mu) + 1e-9)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-10)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-10)
    
    # Compute expected improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Normalize EI to have zero mean and unit std for stable weighting
    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + 1e-10
    ei_norm = (ei - ei_mean) / ei_std
    
    # Normalize cost to [0, 1] range for penalty term
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min + 1e-10
    cost_norm = (cost - cost_min) / cost_range
    
    # Normalize sigma for exploration bonus
    sigma_max = np.max(sigma) + 1e-10
    sigma_norm = sigma / sigma_max
    
    # Cost-aware acquisition: EI - penalty*cost + bonus*sigma
    # This allows high-EI points in expensive regions to be selected
    # if their improvement outweighs the cost, while still preferring
    # cheap points when EI is comparable
    penalty = 0.3
    bonus = 0.2
    
    scores = ei_norm - penalty * cost_norm + bonus * sigma_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
