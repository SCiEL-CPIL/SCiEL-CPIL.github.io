"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state for deadline guard
_bo_start_time = None
_bo_call_count = 0

def acquisition(mu, sigma, f_best):
    global _bo_start_time, _bo_call_count
    
    # Initialize start time on first call
    if _bo_start_time is None:
        _bo_start_time = time.time()
    
    _bo_call_count += 1
    
    # Deadline Guard: 330s limit, use 305s as safety margin
    if time.time() - _bo_start_time > 305:
        # If we are close to timeout, prioritize speed and exploitation
        # to ensure the loop finishes.
        return -mu
    
    # Convert to numpy arrays for safety
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Deterministic Probability of Improvement (PI) acquisition function
    # For minimization: PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # This prioritizes points with high probability of beating the current best
    # while naturally dampening pure noise exploration.
    
    # Handle points with zero or near-zero sigma
    eps = 1e-12
    safe_sigma = np.maximum(sigma, eps)
    
    # Compute the normalized improvement: (f_best - mu) / sigma
    # When f_best < mu, this is negative (low probability of improvement)
    # When f_best > mu, this is positive (high probability of improvement)
    z = (f_best - mu) / safe_sigma
    
    # Probability of Improvement: P(f(x) < f_best) = Phi(z)
    # Use the standard normal CDF
    scores = norm.cdf(z)
    
    # Ensure finite values
    scores[~np.isfinite(scores)] = 0.0
    
    return scores
# EVOLVE-BLOCK-END
