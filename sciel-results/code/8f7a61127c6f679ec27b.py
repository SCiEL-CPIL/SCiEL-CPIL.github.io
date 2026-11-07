"""Acquisition function for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE_MARGIN = 5.0  # seconds before the 330s limit


def _check_deadline():
    """Return True if we should stop to avoid timeout."""
    return (time.time() - _start_time) > (330.0 - _DEADLINE_MARGIN)


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: hybrid of Expected Improvement (EI) and UCB.
    
    For Branin-2D with 30 total evaluations (5 initial + 25 optimization),
    a balanced blend of EI (exploitation) and UCB (exploration) works well.
    We use 70% EI and 30% UCB with beta=0.8 to encourage sufficient
    exploration of the domain to find the global minimum.
    
    Parameters:
        mu: (2000,) array of GP means
        sigma: (2000,) array of GP std devs
        f_best: lowest observed function value
    
    Returns:
        (2000,) array of scores, higher = more desirable
    """
    # Deadline guard: if we're running out of time, return a safe fallback
    if _check_deadline():
        # Return pure EI as a fast fallback
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Standard EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # LCB for minimization: score = -mu + beta * sigma
    # Higher score = more promising (lower predicted value or higher uncertainty)
    # Using a slightly lower beta than UCB since LCB already has stronger exploration
    beta = 1.0
    lcb_score = -mu + beta * sigma_safe
    
    # Normalize both to have similar scale (z-score normalization)
    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + 1e-10
    ei_norm = (ei - ei_mean) / ei_std
    
    lcb_mean = np.mean(lcb_score)
    lcb_std = np.std(lcb_score) + 1e-10
    lcb_norm = (lcb_score - lcb_mean) / lcb_std
    
    # Blend: 60% EI, 40% LCB (more exploratory for Branin's multiple local minima)
    w_ei = 0.6
    w_lcb = 0.4
    score = w_ei * ei_norm + w_lcb * lcb_norm
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
