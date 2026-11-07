"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # Calculate Expected Improvement
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    # Handle sigma == 0
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Calculate EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware acquisition
    # Using sqrt(cost) to reduce the penalty for higher cost points (center)
    # This helps find the global minimum which is at the center (higher cost)
    score = ei / np.sqrt(cost + 1e-9)
    
    return score
# EVOLVE-BLOCK-END
