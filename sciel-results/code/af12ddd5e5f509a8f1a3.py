"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're getting close to the 300s limit, return quickly
    # with a simple cost-aware EI to avoid timeout.
    start_time = time.time()
    
    # Cost-aware Expected Improvement
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Check if we're running low on time (safety margin of 30s)
    if time.time() - start_time > 270:
        return ei / np.maximum(cost, 1e-9)
    
    # Enhanced acquisition: combine EI with exploration and cost awareness
    # Add a small exploration bonus for high-uncertainty regions
    exploration_bonus = 0.1 * sigma_safe / np.maximum(cost, 1e-9)
    
    # Cost-aware score: EI per unit cost, with exploration
    score = ei / np.maximum(cost, 1e-9) + exploration_bonus
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
