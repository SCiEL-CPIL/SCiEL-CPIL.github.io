"""Acquisition function for Bayesian optimization of styblinski tang 1d.
Uses a hybrid acquisition combining expected improvement with a mild
exploration bonus, plus a small exploitation boost near the best prediction.
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
        z = (f_best - mu) / (sigma + 1e-9)
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute z-scores
    z = (f_best - mu) / sigma
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # UCB-style exploration term
    # Use a modest exploration coefficient; beta=1.0 balances exploration/exploitation
    beta = 1.0
    ucb = mu + beta * sigma  # Note: for minimization, we want low mu + high sigma
    # Convert UCB to a score where higher is better (we want to maximize the score)
    # Since we're minimizing, lower UCB is better, so score = -UCB
    ucb_score = -ucb
    
    # Normalize both scores to [0, 1] range for blending
    ei_min, ei_max = np.min(ei), np.max(ei)
    ei_range = ei_max - ei_min + 1e-9
    ei_norm = (ei - ei_min) / ei_range
    
    ucb_min, ucb_max = np.min(ucb_score), np.max(ucb_score)
    ucb_range = ucb_max - ucb_min + 1e-9
    ucb_norm = (ucb_score - ucb_min) / ucb_range
    
    # Blend: 70% EI + 30% UCB
    # This gives strong exploitation with moderate exploration
    w_ei = 0.70
    w_ucb = 0.30
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Add a small exploitation boost: candidates where mu is close to f_best
    # get a small bonus to help refine the neighborhood of the best point
    # This helps when the GP has identified a promising region
    mu_diff = f_best - mu  # positive when mu < f_best (better than best)
    # Scale the boost by how much better mu is than f_best, capped
    boost = np.clip(mu_diff / (sigma + 1e-9), 0, 3.0) * 0.05
    scores += boost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
