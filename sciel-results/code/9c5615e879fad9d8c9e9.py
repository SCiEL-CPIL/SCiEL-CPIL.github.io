"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 300.0
_DEADLINE_MARGIN = 5.0

_F_MIN = -0.8

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Expected Improvement (EI) with cost adjustment.
    
    Parameters:
        mu: array of shape (2000,), GP mean predictions
        sigma: array of shape (2000,), GP standard deviations
        f_best: float, best observed value so far
        cost: array of shape (2000,), evaluation costs
    
    Returns:
        array of shape (2000,), acquisition scores (higher = evaluate next)
    """
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT - _DEADLINE_MARGIN:
        # Return a simple fallback to avoid timeout
        mu = np.asarray(mu, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
        return -mu - 0.1 * (cost - 1.0)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # Higher EI means more expected improvement
    
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # EI can be negative if mu > f_best (we expect to do worse than best)
    # Clip to 0: no improvement expected
    ei = np.maximum(ei, 0.0)
    
    # Cost adjustment: penalize expensive evaluations
    # cost ranges from 1 to 2
    # Normalize: cost_norm in [0, 1]
    cost_norm = (cost - 1.0) / 1.0
    
    # The scale of EI depends on the problem
    # Use a penalty that's proportional to the typical EI scale
    # Estimate typical EI scale from the data
    ei_scale = np.mean(ei[ei > 0]) if np.any(ei > 0) else 0.1
    
    # Penalty weight: small fraction of EI scale
    # We want cost to matter but not dominate
    penalty_weight = 0.15 * ei_scale
    
    # Score: EI minus cost penalty
    score = ei - penalty_weight * cost_norm
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END
