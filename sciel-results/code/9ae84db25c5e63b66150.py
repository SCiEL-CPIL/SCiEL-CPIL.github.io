"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Adaptive UCB acquisition for Ackley 2D minimization.
    
    Strategy:
    - Base score: -mu (exploit low predictions)
    - Exploration: beta * sigma (explore uncertain regions)
    - Below-best bonus: extra weight for points predicted below f_best,
      which reinforces exploitation near the minimum as the search progresses.
    
    The beta parameter adapts: higher exploration early, lower late.
    We estimate progress from how much sigma has shrunk (posterior certainty).
    """
    # Adaptive beta: start high (explore), decrease as sigma shrinks (exploit)
    # Typical sigma range for Ackley 2D with Matern-5/2 ls=0.2: starts ~1-2, drops to ~0.1
    # Use median sigma as a proxy for overall posterior certainty
    med_sigma = np.median(sigma)
    
    # Normalize: high sigma -> high beta (explore), low sigma -> low beta (exploit)
    # Map sigma from [0.05, 1.0] to beta [0.5, 3.0]
    sigma_min = 0.05
    sigma_max = 1.0
    beta_min = 0.5
    beta_max = 3.0
    
    # Clamp med_sigma to range
    med_clamped = np.clip(med_sigma, sigma_min, sigma_max)
    # Linear interpolation
    t = (med_clamped - sigma_min) / (sigma_max - sigma_min)
    beta = beta_min + t * (beta_max - beta_min)
    
    # Base UCB score
    score = -mu + beta * sigma
    
    # Below-best bonus: give extra credit to points predicted below f_best
    # This helps converge to the minimum once we've found a good region
    # Scale the bonus by how far below f_best the prediction is
    below_best = np.maximum(f_best - mu, 0.0)
    # Normalize by sigma to make it relative
    bonus_scale = 0.1
    score += bonus_scale * below_best / (sigma + 1e-9)
    
    return score
# EVOLVE-BLOCK-END
