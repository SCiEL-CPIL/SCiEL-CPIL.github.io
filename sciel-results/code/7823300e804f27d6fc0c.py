"""Structural change: use Probability of Improvement (PI) with adaptive cost weighting."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state for deadline guard (shared across calls in the loop)
_START_TIME = time.time()
_DEADLINE = _START_TIME + 300.0  # 300s wall-clock limit per evaluation; use safety margin

def _elapsed_ok():
    """Return True if we have at least ~5s margin before the hard limit."""
    return (time.time() < _DEADLINE - 5.0)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Probability of Improvement (PI) with adaptive cost weighting.
    
    mu, sigma, f_best, cost: each shape (2000,)
    f_best: scalar (lowest observed value)
    cost: per-candidate cost = 1 + mean(x_scaled)
    
    Returns: 2000 finite scores (higher = evaluate next)
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against zero/negative cost
    cost_safe = np.maximum(cost, 1e-9)
    
    # Robust sigma: ensure it's positive and not too small
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Probability of Improvement (PI): P(f(x) < f_best)
    # For Gaussian f(x) ~ N(mu, sigma^2), P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    
    # Make PI non-negative (numerical safety)
    pi = np.maximum(pi, 0.0)
    
    # Adaptive cost weighting: use a power-law cost penalty instead of linear division
    # This allows more exploration of slightly more expensive points when they have high PI
    # cost_weight = cost_safe^alpha, where alpha < 1 softens the penalty
    alpha = 0.7  # Soften cost penalty: alpha=1 is linear, alpha<1 is softer
    
    # Normalize cost to [0, 1] for stability
    c_min = np.min(cost_safe)
    c_max = np.max(cost_safe)
    c_range = c_max - c_min
    if c_range > 1e-12:
        cost_norm = (cost_safe - c_min) / c_range
    else:
        cost_norm = np.ones_like(cost_safe)
    
    # Apply power-law weighting: score = PI / (cost_norm + epsilon)^alpha
    # This is equivalent to PI * cost_norm^(-alpha)
    epsilon = 1e-6
    cost_weight = (cost_norm + epsilon) ** alpha
    score = pi / cost_weight
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
