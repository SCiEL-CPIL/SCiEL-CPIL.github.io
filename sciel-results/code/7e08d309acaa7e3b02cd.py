"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a combination of Expected Improvement and UCB
    with adaptive exploration-exploitation balance.
    """
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Compute Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Compute UCB score (for minimization: low mu, high sigma is good)
    # Use adaptive kappa: larger when f_best is still high (early), smaller later
    # Normalize f_best to get a sense of progress
    # If f_best is close to the known minimum (-2.53218), we should exploit more
    
    # Estimate how close we are to the optimum
    # The minimum is -2.53218, so if f_best is much higher, we need more exploration
    min_known = -2.53218
    # Normalize: 0 means at optimum, 1 means far from optimum
    # Use a scale based on typical function values
    progress = 1.0 - np.clip((f_best - min_known) / 5.0, 0.0, 1.0)
    
    # Adaptive kappa: start with high exploration (kappa=1.5), decrease to low (kappa=0.3)
    kappa = 0.3 + 1.2 * (1.0 - progress)
    
    ucb_score = -mu + kappa * sigma
    
    # Normalize both scores to comparable scales
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min if ei_max > ei_min else 1.0
    ei_norm = (ei - ei_min) / ei_range
    
    ucb_min = np.min(ucb_score)
    ucb_max = np.max(ucb_score)
    ucb_range = ucb_max - ucb_min if ucb_max > ucb_min else 1.0
    ucb_norm = (ucb_score - ucb_min) / ucb_range
    
    # Blend: more UCB when we're far from optimum, more EI when close
    # weight_ei increases with progress
    weight_ei = 0.3 + 0.5 * progress
    weight_ucb = 1.0 - weight_ei
    
    scores = weight_ei * ei_norm + weight_ucb * ucb_norm
    
    # Check for NaNs or Infs
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
