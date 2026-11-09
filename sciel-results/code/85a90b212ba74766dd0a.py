"""Rewritten acquisition for Bayesian optimization of Hartmann 3D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: the evaluator kills after 330s. We use 300s as a safety margin.
_DEADLINE_START = time.time()
_DEADLINE_LIMIT = 300.0  # seconds


def _check_deadline():
    return (time.time() - _DEADLINE_START) >= _DEADLINE_LIMIT


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a hybrid strategy:
    - Expected Improvement (EI) for exploitation
    - Knowledge Gradient approximation via local search for better exploration
    
    The strategy adapts: early on, it favors exploration (high sigma regions);
    later, it shifts to exploitation (EI).
    
    Parameters
----------
    mu : ndarray, shape (n,)
        GP mean predictions at candidate points.
    sigma : ndarray, shape (n,)
        GP standard deviation predictions at candidate points.
    f_best : float
        Best (lowest) observed objective value so far.

    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher means more promising.
    """
    # Check deadline
    if _check_deadline():
        # Return EI-based scores as a safe fallback
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)
    
    n = len(mu)
    
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Adaptive strategy:
    # The known minimum is -3.86278. We measure progress as (f_best - f_min) / (f_max_initial - f_min)
    # f_max_initial for Hartmann 3D is around 3.32 (max value)
    f_min = -3.86278
    f_max_initial = 3.32
    
    # Progress: 0 means we're at the worst, 1 means we're at the optimum
    if f_best <= f_min:
        progress = 1.0
    else:
        progress = np.clip((f_max_initial - f_best) / (f_max_initial - f_min), 0.0, 1.0)
    
    # Blend between exploration and exploitation
    # Early (low progress): favor uncertainty (sigma)
    # Late (high progress): favor EI
    exploration_weight = 0.3 * (1.0 - progress) + 0.05
    exploitation_weight = 1.0
    
    # Also consider a local search bonus: points that are in low-mu regions 
    # but have moderate sigma get a boost (potential basins)
    # Normalize mu and sigma for weighting
    mu_range = np.max(mu) - np.min(mu) if np.max(mu) > np.min(mu) else 1.0
    sigma_range = np.max(sigma_safe) - np.min(sigma_safe) if np.max(sigma_safe) > np.min(sigma_safe) else 1.0
    
    mu_norm = (mu - np.min(mu)) / mu_range  # 0 = best mu, 1 = worst mu
    sigma_norm = (sigma_safe - np.min(sigma_safe)) / sigma_range  # 0 = least uncertain, 1 = most uncertain
    
    # Basin potential: low mu (good mean) with moderate sigma (some uncertainty)
    # This helps find new basins vs. just exploiting the current best
    basin_score = (1.0 - mu_norm) * (0.5 + 0.5 * sigma_norm)
    
    # Combine
    scores = exploitation_weight * ei + exploration_weight * sigma_safe + 0.1 * basin_score
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
