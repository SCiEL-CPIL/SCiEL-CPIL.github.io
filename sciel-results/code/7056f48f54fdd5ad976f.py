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
    
    # Smooth exploration bonus: proportional to sigma but saturates
    # This encourages exploring uncertain regions without the discontinuity of max
    exploration = sigma / (1.0 + sigma)
    
    # Combine: EI dominates, exploration is a small additive term
    score = ei + 0.1 * exploration
    
    # Cost-aware normalization
    score = score / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
