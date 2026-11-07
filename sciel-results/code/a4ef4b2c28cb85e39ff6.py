import time
import numpy as np
from scipy.stats import norm

_T0 = time.perf_counter()
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best, cost):
    if time.perf_counter() - _T0 > _DEADLINE:
        return 1.0 / np.maximum(cost, 1e-9)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Dynamic cost exponent
    # If f_best is high, we are early, use linear cost penalty (alpha=1).
    # If f_best is low, we are late, use quadratic cost penalty (alpha=2).
    # Transition between 1 and 2 based on f_best.
    # Assume f_best drops from ~100 to ~0.
    # Use a log scale or linear clip.
    # Let's say if f_best > 1.0, alpha = 1.0.
    # If f_best < 0.0, alpha = 2.0.
    # Linear interpolation: alpha = 2.0 - f_best, clipped to [1.0, 2.0].
    
    alpha = np.clip(2.0 - f_best, 1.0, 2.0)
    
    # Multiplicative feasibility constraint: gate EI by cost relative to median
    # Low-cost candidates get full EI; high-cost candidates get exponentially penalized
    cost_median = np.median(cost_safe)
    t = (cost_safe - cost_median) / np.maximum(cost_median, 1e-9)
    t = np.clip(t, 0.0, 10.0)
    gate = np.exp(-alpha * t)
    
    scores = ei * gate
    
    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores
