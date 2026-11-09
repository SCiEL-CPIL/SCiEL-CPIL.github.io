"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() - _start_time > 320:
        return np.zeros_like(mu)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Hybrid: combine EI and PI to balance exploitation and exploration
    # Use a weighted combination that adapts to the cost landscape
    # When cost is high, we need stronger exploitation (EI)
    # When cost is low, we can afford more exploration (PI)
    cost_norm = cost / (np.max(cost) + 1e-9)
    weight_ei = 0.5 + 0.5 * cost_norm  # Higher weight on EI for expensive regions
    weight_pi = 1.0 - weight_ei
    
    combined = weight_ei * ei + weight_pi * pi * (f_best - mu)
    
    # Cost-aware score
    scores = combined / (cost + 1e-9)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
