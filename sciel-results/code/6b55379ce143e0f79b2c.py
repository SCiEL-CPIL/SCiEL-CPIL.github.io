"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_call_count = 0
_start_time = None

def acquisition(mu, sigma, f_best):
    global _call_count, _start_time
    
    if _start_time is None:
        _start_time = time.time()
    _call_count += 1
    
    # Deadline guard: if we're close to the 330s limit, return a safe fallback
    elapsed = time.time() - _start_time
    if elapsed > 325:
        # Return a simple EI to finish quickly
        z = (f_best - mu) / (sigma + 1e-9)
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Hybrid acquisition: combine EI with UCB-like exploration
    # EI for minimization: (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Normalize EI to [0, 1] roughly for combination
    ei_max = np.max(ei)
    if ei_max > 1e-9:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    # UCB component for exploration
    # Scale sigma to be comparable to EI
    sigma_norm = sigma / (np.max(sigma) + 1e-9)
    
    # Adaptive weight: start with more exploration, shift to exploitation
    # For 20 acquisition calls, transition from exploration to exploitation
    if _call_count <= 3:
        w_explore = 0.7
    elif _call_count <= 6:
        w_explore = 0.5
    elif _call_count <= 10:
        w_explore = 0.3
    elif _call_count <= 15:
        w_explore = 0.2
    else:
        w_explore = 0.1
    
    # Combine: higher score means more likely to be evaluated
    # We want high EI (exploitation) and high sigma (exploration)
    scores = (1 - w_explore) * ei_norm + w_explore * sigma_norm
    
    # Add a small bonus for points with low mu (likely good)
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    if mu_max - mu_min > 1e-9:
        mu_norm = (mu_max - mu) / (mu_max - mu_min)
        scores += 0.1 * mu_norm
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
