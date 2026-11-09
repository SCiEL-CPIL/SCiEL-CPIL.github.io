"""
Cost-aware acquisition using a fixed blend of Expected Improvement (EI) and Uncertainty (UCB), normalized by cost.
This avoids adaptive parameter tuning which can be unstable, and directly optimizes for improvement per unit cost.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state for deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 300.0

def _time_remaining():
    """Return seconds remaining before hard deadline."""
    return _DEADLINE - time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for Bayesian optimization.
    
    mu, sigma: shape (2000,), GP mean and std dev in objective units
    f_best: scalar, lowest observed value
    cost: shape (2000,), per-candidate cost = 1 + mean(x_scaled)
    
    Returns: 2000 finite scores (higher = evaluate next)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure cost is at least 1 to avoid division by zero or huge spikes
    cost_safe = np.maximum(cost, 1.0)
    
    # Safety margin: if we are close to the deadline, return a simple proxy
    # -mu/cost is a fast, reasonable heuristic for exploitation
    if _time_remaining() < 2.0:
        score = -mu / cost_safe
        return np.where(np.isfinite(score), score, 0.0)

    # Numerical stability for sigma
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Compute Expected Improvement (EI) for minimization
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    # Clip z to prevent numerical issues in norm.pdf/cdf
    z = np.clip(z, -10.0, 10.0)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Blend EI with UCB term (sigma) to encourage exploration
    # beta controls the exploration-exploitation trade-off
    # A fixed beta=1.0 is robust for this cost-aware setting
    beta = 1.0
    
    # Combined acquisition value before cost normalization
    # We add beta*sigma to EI to ensure high-uncertainty regions are considered
    raw_score = ei + beta * sigma_safe
    
    # Normalize by cost to prioritize cheaper candidates
    # This effectively maximizes "improvement per dollar"
    score = raw_score / cost_safe
    
    # Ensure finite and non-negative
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
