import numpy as np
from scipy.stats import norm
import time

# Global variable to track start time for deadline guard
_start_time = None

def acquisition(mu, sigma, f_best):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we are too close to the limit, return a simple score
    # 330s total limit. Let's say we want to be done by 325s.
    elapsed = time.time() - _start_time
    if elapsed > 325.0:
        # Return a simple exploitation score: -mu
        # This ensures we pick the best current point if we are running out of time
        return -np.asarray(mu, dtype=np.float64)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Probability of Improvement (PoI)
    # PoI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # This focuses on the likelihood of beating the current best, which better
    # distributes exploration across multiple global minima compared to EI.
    # EI can be dominated by high-uncertainty regions that are unlikely to improve,
    # whereas PoI directly measures the chance of improvement.
    
    z = (f_best - mu) / sigma_safe
    poi = norm.cdf(z)
    
    # To ensure finite scores
    scores = np.where(np.isfinite(poi), poi, 0.0)
    
    return scores
