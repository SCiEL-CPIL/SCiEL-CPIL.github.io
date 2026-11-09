"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    t0 = time.time()
    
    # Clamp sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Standard EI for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a very small exploration term to avoid over-concentration
    # This is much smaller than typical exploration bonuses
    exploration = 1e-6 * sigma_safe
    
    score = ei + exploration
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline check
    if time.time() - t0 > 290.0:
        return score
    
    return score
# EVOLVE-BLOCK-END
