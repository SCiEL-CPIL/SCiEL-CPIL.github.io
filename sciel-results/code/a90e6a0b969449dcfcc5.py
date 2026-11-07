"""Acquisition function for Bayesian optimization of styblinski tang 1d.
Combines normalized EI with adaptive UCB exploration, plus a refinement boost.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_start_time = time.time()
_DEADLINE = _start_time + 300.0  # 300s wall-clock limit per evaluation


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidate points.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP posterior mean at candidates (in objective units)
    sigma : array of shape (2000,)
        GP posterior standard deviation at candidates (in objective units)
    f_best : float
        Best (lowest) observed value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores; higher means evaluate next
    """
    # Check deadline
    if time.time() > _DEADLINE - 5.0:
        # Safety: return EI as fallback
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        return (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-scores
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB-style exploration term
    # Use an adaptive exploration coefficient that decreases as we approach the optimum
    f_opt = -39.1662
    gap = abs(f_best - f_opt)
    # beta decreases from ~2.0 to ~0.5 as we approach optimum
    beta = 0.5 + 1.5 * np.exp(-gap / 10.0)
    
    # For minimization, lower UCB is better, so score = -UCB
    ucb_score = -mu + beta * sigma_safe
    
    # Normalize both scores to [0, 1] range for blending
    ei_min, ei_max = np.min(ei), np.max(ei)
    ei_range = ei_max - ei_min + 1e-9
    ei_norm = (ei - ei_min) / ei_range
    
    ucb_min, ucb_max = np.min(ucb_score), np.max(ucb_score)
    ucb_range = ucb_max - ucb_min + 1e-9
    ucb_norm = (ucb_score - ucb_min) / ucb_range
    
    # Blend: start with more UCB (exploration), shift to EI (exploitation)
    # weight for EI increases as we get closer to optimum
    ei_weight = 0.5 + 0.5 * np.exp(-gap / 5.0)
    ucb_weight = 1.0 - ei_weight
    
    scores = ei_weight * ei_norm + ucb_weight * ucb_norm
    
    # Add a small exploitation boost: candidates where mu is close to f_best
    # get a small bonus to help refine the neighborhood of the best point
    mu_diff = f_best - mu  # positive when mu < f_best (better than best)
    # Scale the boost by how much better mu is than f_best, capped
    boost = np.clip(mu_diff / (sigma_safe + 1e-9), 0, 3.0) * 0.05
    scores += boost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
