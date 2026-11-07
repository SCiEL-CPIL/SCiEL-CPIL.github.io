"""Rewritten program with Cost-Aware Expected Improvement acquisition."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global timer for deadline guard
_start_time = time.time()
_TIME_LIMIT = 315.0  # 15s margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we are too close to the limit, return a safe fallback
    if time.time() - _start_time > _TIME_LIMIT:
        return np.zeros(len(mu))
    
    # Cost-Aware Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    # For minimization, we want to maximize the expected improvement over f_best
    # Then divide by cost to account for evaluation cost
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the CDF of standard normal and phi is the PDF
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (should be by construction, but clip for safety)
    ei = np.maximum(ei, 0.0)
    
    # Ensure cost is positive to avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    
    # Cost-aware score: EI / cost
    scores = ei / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
