"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    t0 = time.time()
    
    # Clamp sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # UCB for minimization: maximize -mu + beta * sigma
    # beta controls the exploration-exploitation trade-off
    beta = 1.0
    score = -mu + beta * sigma_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    # Deadline check
    if time.time() - t0 > 290.0:
        return score
    
    return score
# EVOLVE-BLOCK-END
