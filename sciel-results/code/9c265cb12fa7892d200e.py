"""Cost-aware acquisition using log-domain stable EI/cost score."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 320s wall-clock safety margin (limit is 330s)

def _elapsed_ok():
    return (time.time() - _START_TIME) < (_DEADLINE - 5.0)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for Bayesian optimization.
    
    mu, sigma, f_best, cost: each shape (2000,)
    f_best: scalar (lowest observed value)
    cost: per-candidate cost = 1 + mean(x_scaled)
    
    Returns: 2000 finite scores (higher = evaluate next)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against zero/negative values
    cost_safe = np.maximum(cost, 1e-9)
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement (EI)
    diff = f_best - mu
    z = diff / sigma_safe
    
    # Compute EI in a numerically stable way
    # For large positive z, EI ~ diff; for large negative z, EI ~ 0
    # Use the standard formula but clip z to avoid overflow in pdf
    z_clipped = np.clip(z, -30.0, 30.0)
    ei = diff * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost to get improvement per unit cost
    score = ei / cost_safe
    
    # Ensure all scores are finite and non-negative
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
