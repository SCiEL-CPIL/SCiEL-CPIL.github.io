import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0  # 5 seconds before the 330s limit


def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        return np.zeros_like(mu)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-6)
    
    # Cost-aware Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Bounded exploration bonus: sigma/(sigma + 0.1) saturates near 1 for large sigma
    # but is much smaller than sigma for small sigma, preventing exploration from dominating
    exploration = sigma_safe / (sigma_safe + 0.1)
    
    # Combine: EI dominates, exploration is a moderate additive term
    score = ei + 0.15 * exploration
    
    # Smooth cost-aware normalization: 1/(1+cost) penalizes expensive points gently
    # instead of the harsh 1/cost which can create discontinuities
    cost_factor = 1.0 / (1.0 + cost_safe)
    score = score * cost_factor
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
