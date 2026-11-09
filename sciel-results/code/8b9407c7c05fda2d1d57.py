"""Rewritten acquisition for Bayesian optimization of 6d Hartmann."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_acq_start_time = None


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement with an adaptive exploration bonus.
    
    Parameters:
    -----------
    mu : ndarray, shape (n,)
        Posterior mean at candidate points.
    sigma : ndarray, shape (n,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    global _acq_start_time
    if _acq_start_time is None:
        _acq_start_time = time.time()
    
    # Deadline guard: if we're running too long, return a simple fallback
    elapsed = time.time() - _acq_start_time
    if elapsed > 280:
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (for minimization)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Adaptive exploration weight:
    # Scale exploration by how uncertain the posterior is relative to the
    # remaining improvement potential. Early on (high sigma / low mu_range),
    # explore more; later (low sigma / high mu_range), exploit more.
    mu_min = np.min(mu)
    improvement_scale = max(f_best - mu_min, 1e-9)
    sigma_mean = np.mean(sigma_safe)
    sigma_rel = sigma_mean / improvement_scale
    # Clamp to [0, 1] and map to weight in [0.05, 0.18]
    sigma_rel_clamped = np.clip(sigma_rel, 0.0, 1.0)
    exploration_weight = 0.05 + 0.13 * sigma_rel_clamped
    
    exploration_bonus = exploration_weight * sigma_safe
    
    # Combined score
    scores = ei + exploration_bonus
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
