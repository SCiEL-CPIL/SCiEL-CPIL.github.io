# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Beale 2D function.
    Uses a combination of Expected Improvement and a local improvement bonus
    to balance exploration and exploitation.
    """
    # Handle numerical stability
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Probability of improvement
    prob_improve = norm.cdf(z)
    
    # A "regret" term that rewards points where mu is already below f_best
    # This encourages exploitation of known good regions
    # We use a soft threshold: if mu is close to or below f_best, give a bonus
    # Scale: (f_best - mu) is the potential improvement
    # We want to strongly favor points where mu << f_best
    
    # Local improvement: points where the model predicts a lower value
    # Use a Gaussian kernel centered at f_best to create a "basin of attraction"
    # This is a structural change: instead of linear UCB, use a nonlinear
    # bonus that peaks when mu is significantly below f_best
    
    # Normalize mu relative to f_best
    # When mu < f_best, we're in a promising region
    # Use a tanh-like function to smoothly transition
    delta = f_best - mu  # positive when mu < f_best (improvement)
    
    # Scale delta to create a smooth bonus
    # sigma gives us a natural scale
    bonus = np.tanh(delta / (sigma_safe + 1e-10))
    
    # Combine: EI provides the expected improvement, bonus reinforces exploitation
    # When f_best is large (early), we want more exploration
    # When f_best is small (late), we want more exploitation
    
    # Adaptive weighting based on f_best magnitude
    # f_best starts high and decreases toward 0
    if f_best > 100:
        w_ei = 0.8
        w_bonus = 0.2
    elif f_best > 10:
        w_ei = 0.6
        w_bonus = 0.4
    elif f_best > 1:
        w_ei = 0.4
        w_bonus = 0.6
    else:
        w_ei = 0.2
        w_bonus = 0.8
    
    # Normalize EI to [0, 1]
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ei_min = np.min(ei)
    ei_range = ei_max - ei_min
    if ei_range > 1e-10:
        ei_norm = (ei - ei_min) / ei_range
    else:
        ei_norm = np.zeros_like(ei)
    
    # Bonus is already in [-1, 1], shift to [0, 1]
    bonus_norm = (bonus + 1) / 2
    
    score = w_ei * ei_norm + w_bonus * bonus_norm
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
