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
    
    # Ensure sigma is positive and reasonably large to avoid numerical issues
    # Although PI is robust, extremely small sigma can cause z to be huge,
    # but norm.cdf handles inf correctly (returns 0 or 1).
    # We clip sigma to a small positive value to prevent division by zero in z calculation.
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Cost is in [1, 2] based on the problem description (1 + mean(x scaled to [0,1]))
    cost_safe = np.maximum(cost, 1e-8)
    
    # Calculate z-score for Probability of Improvement
    # For minimization, improvement is f(x) < f_best
    # z = (f_best - mu) / sigma
    # P(f(x) < f_best) = Phi(z)
    
    z = (f_best - mu) / sigma_safe
    
    # Calculate PI
    pi = norm.cdf(z)
    
    # Cost-aware score: PI / cost
    # Higher score means higher probability of improvement per unit cost
    scores = pi / cost_safe
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
