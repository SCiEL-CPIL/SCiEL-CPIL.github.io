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
    
    # Use Probability of Improvement (PoI) with a small sigma-weighted exploration term
    # PoI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # This decouples from improvement magnitude, favoring broad coverage of multi-modal landscapes
    
    # Calculate z-score for PoI
    z = (f_best - mu) / sigma_safe
    
    # Probability of improvement
    poi = norm.cdf(z)
    
    # Add a small exploration term weighted by sigma to encourage diversity
    # This helps find different basins in the multi-modal Branin landscape
    # Use a small coefficient to not dominate the PoI signal
    exploration_term = 0.1 * sigma_safe
    
    scores = poi + exploration_term
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
