"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time to track elapsed time for deadline guard
_START_TIME = None

def acquisition(mu, sigma, f_best, cost):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Deadline guard: If we are getting close to the 330s limit, 
    # we should return a safe, quick result. 
    # Since this function is called many times, checking time every time is cheap.
    # However, the heavy lifting is outside. 
    # If the caller is stuck, we can't help much, but we can ensure we don't add latency.
    
    # Ensure inputs are numpy arrays for vectorized ops
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and numerical instability
    sigma_safe = np.clip(sigma, 1e-8, None)
    
    # Calculate z-score
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid extreme values in norm.cdf/pdf
    # norm.cdf(-10) ~ 0, norm.pdf(-10) ~ 0
    # norm.cdf(10) ~ 1, norm.pdf(10) ~ 0
    z_clipped = np.clip(z, -10, 10)
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z_clipped)
    pdf_z = norm.pdf(z_clipped)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Cost-aware penalty
    # Using exponential suppression as per inspiration
    # lambda = 1.0 provides a sharp trade-off
    lambda_cost = 1.0
    penalty = np.exp(-lambda_cost * cost)
    
    scores = ei * penalty
    
    # Ensure all scores are finite
    # If any are NaN or Inf, replace with 0 (neutral score)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
