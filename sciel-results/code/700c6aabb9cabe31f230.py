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
    
    # GP-UCB with fixed beta for exploration
    # Use a moderate beta value to balance exploration and exploitation
    beta = 2.0
    ucb = -mu + beta * sigma_safe  # For minimization, we want low mu and high sigma
    
    # Combine EI and UCB: use a weighted combination
    # Normalize both to comparable scales
    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + 1e-12
    ucb_mean = np.mean(ucb)
    ucb_std = np.std(ucb) + 1e-12
    
    ei_norm = (ei - ei_mean) / ei_std
    ucb_norm = (ucb - ucb_mean) / ucb_std
    
    # Weighted combination: 0.6 EI + 0.4 UCB
    score = 0.6 * ei + 0.4 * ucb
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline check
    if time.time() - t0 > 290.0:
        return score
    
    return score
# EVOLVE-BLOCK-END
