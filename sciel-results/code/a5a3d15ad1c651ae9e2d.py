"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: kill after 320s wall-clock
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Args:
        mu: (2000,) array of GP means
        sigma: (2000,) array of GP standard deviations
        f_best: float, best observed value so far
        cost: (2000,) array of costs for each candidate
    
    Returns:
        (2000,) array of acquisition values (higher = more promising)
    """
    # Deadline guard: if we're getting close to the limit, return a simple fallback
    if time.time() > _DEADLINE:
        return np.full_like(mu, 0.0)
    
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost
    # Use a moderate cost penalty to prefer cheaper evaluations
    # but not as aggressive as 1.5 which might hurt exploration
    scores = ei / cost_safe
    
    # Add a small exploration term to encourage visiting high-uncertainty regions
    # This helps escape local optima and explore the space more broadly
    # Scale exploration by sigma to focus on high-uncertainty areas
    exploration = 0.05 * sigma_safe / cost_safe
    
    scores = scores + exploration
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
