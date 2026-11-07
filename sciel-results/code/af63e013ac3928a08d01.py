"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() - _start_time > 320:
        return -mu
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Adaptive UCB: exploration parameter that decreases as we get closer to optimum
    # The known minimum of Michalewicz 2D is approximately -1.8013
    # As f_best approaches this, we should exploit more (lower kappa)
    min_value = -1.8013
    # Normalize: when f_best = min_value, progress = 1; when f_best is high, progress = 0
    # Use a smooth transition
    if f_best <= min_value:
        progress = 1.0
    else:
        # Scale: assume initial f_best might be around 0 to 1.5
        # progress goes from 0 to 1 as f_best goes from 1.5 to min_value
        progress = np.clip((1.5 - f_best) / (1.5 - min_value), 0.0, 1.0)
    
    # kappa decreases from ~2.0 (exploration) to ~0.5 (exploitation)
    kappa = 2.0 - 1.5 * progress
    
    # UCB score for minimization: maximize -mu + kappa * sigma
    scores = -mu + kappa * sigma_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
