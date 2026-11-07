import numpy as np
from scipy.stats import norm
import time

_start_time = None
_deadline = 300.0
_safety_margin = 5.0

def _check_deadline():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    if time.time() > _start_time + _deadline - _safety_margin:
        return True
    return False

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # f_best is scalar
    # z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (minimization)
    phi_z = norm.pdf(z)
    cdf_z = norm.cdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * phi_z
    
    # Cost-aware score
    # Cost is in [1, 2]
    # Using EI / Cost
    cost_safe = np.maximum(cost, 1e-8)
    
    scores = ei / cost_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
