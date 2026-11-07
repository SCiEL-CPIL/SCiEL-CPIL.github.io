"""Probability of Improvement with adaptive threshold for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    n = len(mu)
    
    # Deadline guard
    start_time = time.time()
    deadline = 280.0
    
    # Clamp sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Adaptive improvement threshold: scales with current best
    # When f_best is large, we demand larger relative improvements
    # When f_best is small (near optimum), we demand smaller absolute improvements
    # Use a floor to avoid zero threshold
    xi = max(0.05 * f_best, 1e-4)
    
    # Compute z-score: (f_best - xi - mu) / sigma
    # This is the standardized distance from the current best (minus threshold) to the predicted mean
    z = (f_best - xi - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best - xi) = Phi(z)
    # where Phi is the standard normal CDF
    scores = norm.cdf(z)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        # Fallback: return scores based on negative mean (exploit)
        return -mu
    
    return scores
# EVOLVE-BLOCK-END
