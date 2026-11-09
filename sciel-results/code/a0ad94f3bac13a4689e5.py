"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Deadline guard
    deadline = start_time + 290
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement (PI)
    pi = norm.cdf(z)
    
    # Expected Improvement (EI) - captures magnitude of expected improvement
    ei = sigma_safe * (z * norm.cdf(z) + norm.pdf(z))
    ei = np.maximum(ei, 0.0)  # EI is non-negative
    
    # Normalize both PI and EI to [0, 1] using min-max
    def normalize(arr):
        arr_min = np.min(arr)
        arr_max = np.max(arr)
        arr_range = arr_max - arr_min
        if arr_range < 1e-9:
            return np.ones_like(arr)
        return (arr - arr_min) / arr_range
    
    pi_norm = normalize(pi)
    ei_norm = normalize(ei)
    
    # Blend: 60% normalized EI (captures improvement magnitude) + 40% normalized PI
    blended = 0.6 * ei_norm + 0.4 * pi_norm
    
    # Cost-aware scoring: divide by cost to penalize expensive evaluations
    # Cost is in [1, 2], so this provides mild cost awareness
    score = blended / cost
    
    # Add small exploration term to avoid premature convergence
    # Use inverse of cost as a proxy for "cheapness" to encourage exploration in cheap regions
    exploration = 0.05 * (1.0 / cost)
    
    score = score + exploration
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        return blended
    
    return score
# EVOLVE-BLOCK-END
