"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = None

def acquisition(mu, sigma, f_best):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we're running long, return a simple score
    elapsed = time.time() - _start_time
    if elapsed > 300:
        # Fallback: simple UCB
        return -mu + 1.0 * sigma
    
    # Hybrid acquisition: blend EI with exploration
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Probability of Improvement (more exploration-friendly)
    poi = norm.cdf(z)
    
    # UCB component for exploration
    ucb = -mu + 1.2 * sigma_safe
    
    # Blend: primarily EI with some POI and UCB for multimodal exploration
    # Scale components to be comparable
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ucb_max = np.max(ucb) if np.max(ucb) > 0 else 1.0
    
    score = 0.6 * ei + 0.2 * (poi * ei_max) + 0.2 * (ucb / ucb_max) * ei_max
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
