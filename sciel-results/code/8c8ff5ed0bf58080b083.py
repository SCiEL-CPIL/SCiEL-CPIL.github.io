"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 320.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a cost-adjusted Expected Improvement
    with an adaptive multiplicative cost penalty and exploration term.
    
    mu, sigma, f_best, cost: arrays of shape (2000,)
    Returns: array of 2000 finite scores (higher = better candidate to evaluate)
    """
    # Deadline guard: if we've spent too much time, return a simple fallback
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        return np.zeros(len(mu), dtype=np.float64)

    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip EI to be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Adaptive multiplicative cost penalty:
    # Use cost^gamma where gamma adapts based on f_best.
    # Early on (high f_best), use smaller gamma to allow exploration of expensive regions.
    # Late on (low f_best), use larger gamma to conserve budget.
    # f_best starts around 1-2 and decreases toward 0.
    # Map f_best from [0, 2] to gamma in [0.5, 1.5]
    f_best_clamped = np.clip(f_best, 0.0, 2.0)
    gamma = 0.5 + 0.5 * f_best_clamped  # gamma in [0.5, 1.5]
    
    # cost is in [1, 2], so cost^gamma is in [1, 2^gamma]
    cost_penalty = np.power(cost, gamma)
    
    # Exploration term: weight sigma to encourage uncertainty
    # Use a small beta to balance exploration and exploitation
    beta = 0.1
    
    # Combined score: (EI + beta * sigma) / cost_penalty
    scores = (ei + beta * sigma_safe) / cost_penalty
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
