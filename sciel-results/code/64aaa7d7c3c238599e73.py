"""Acquisition function for Bayesian optimization of styblinski tang 1d.
Uses Expected Improvement with a mild exploration component derived from
the GP posterior, tuned for the 30-evaluation budget on the 1D domain.
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
    # Deadline safety check
    if time.time() > _DEADLINE - 5.0:
        # Fallback: simple EI
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        return (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (for minimization)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Exploration: UCB-like term with dynamic beta based on f_best.
    # As f_best improves (more negative), reduce exploration.
    # Map f_best from [0, -39] to t in [1, 0], then beta from [1.0, 0.3]
    t = np.clip((f_best + 39.0) / 39.0, 0.0, 1.0)
    beta = 0.3 + 0.7 * t
    
    # Scale sigma to be comparable to EI magnitude
    ucb = beta * sigma_safe / 10.0
    
    # Small exploitation boost for points predicted below f_best
    mu_below = np.maximum(f_best - mu, 0.0)
    exploit_boost = np.clip(mu_below / 40.0, 0.0, 1.0) * 0.1
    
    # Combine
    scores = ei + ucb + exploit_boost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
