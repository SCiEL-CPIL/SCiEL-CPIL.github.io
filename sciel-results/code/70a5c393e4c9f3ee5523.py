import numpy as np
from scipy.stats import norm
import time

_start_time = None
_DEADLINE = 320.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Fallback: Simple, fast score
        # Prefer low mu and low cost
        cost_safe = np.maximum(cost, 1e-9)
        return -mu / cost_safe

    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Dynamically relaxed threshold for meaningful improvement
    # This focuses on gains that are not just noise but actual progress
    threshold = f_best - 0.05
    
    # Calculate z-score relative to the relaxed threshold
    # z = (threshold - mu) / sigma
    # If mu < threshold, z > 0 (high probability of beating the threshold)
    z = (threshold - mu) / sigma_safe
    
    # Probability of Improvement (POI)
    # POI = P(f(x) < threshold) = Phi(z)
    # This is a bounded [0, 1] value, making it stable and exploitative
    poi = norm.cdf(z)
    
    # Cost-aware score: POI per unit cost
    # We want to maximize the probability of finding a point better than the threshold
    # per unit cost. This prioritizes cheap evaluations with high confidence.
    score = poi / cost_safe
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
