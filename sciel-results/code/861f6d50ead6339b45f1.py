"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


_start_time = time.time()


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: return a safe fallback if we're getting close to the limit
    elapsed = time.time() - _start_time
    if elapsed > 270:
        # Simple EI / cost as fallback
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        z = np.clip(z, -30, 30)
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return ei / np.maximum(cost, 1e-9)
    
    # Cost-aware acquisition combining EI and UCB-like exploration
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -30, 30)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Uncertainty bonus (exploration)
    # Weight uncertainty to balance exploration vs exploitation
    # The cost term penalizes expensive points
    # We use a hybrid: EI/cost + lambda * sigma/cost
    # where lambda controls exploration
    
    # Adaptive lambda based on how much we've explored
    # Early on, explore more; later, exploit more
    # We don't have explicit iteration count, but f_best gives some signal
    # Use a fixed reasonable lambda that works well for Hartmann-3D
    
    lambda_explore = 1.5
    
    # Cost-aware score: combine EI and uncertainty, normalized by cost
    score = (ei + lambda_explore * sigma_safe) / np.maximum(cost, 1e-9)
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
