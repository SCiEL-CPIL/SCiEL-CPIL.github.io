"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Pre-compute constants for speed
_EPS = 1e-9
_NORM_CDF = norm.cdf
_NORM_PDF = norm.pdf


def acquisition(mu, sigma, f_best, cost):
    """
    Expected Improvement per unit cost, with a small exploration bonus.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of GP predictions at candidate points.
    sigma : array-like, shape (2000,)
        Standard deviation of GP predictions at candidate points.
    f_best : float
        Best (lowest) observed objective value.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate next).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, _EPS)
    cost_safe = np.maximum(cost, _EPS)
    
    # Standard Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * _NORM_CDF(z) + sigma_safe * _NORM_PDF(z)
    
    # Cost-aware: EI per unit cost
    cost_aware_ei = ei / cost_safe
    
    # Small exploration bonus to help escape local optima in multimodal landscape
    exploration_bonus = 0.05 * sigma_safe / cost_safe
    
    scores = cost_aware_ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
