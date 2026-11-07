"""Bayesian optimization acquisition for Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

_obs_count = 5


def acquisition(mu, sigma, f_best):
    """
    Combined GP-UCB and normalized Expected Improvement acquisition function.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP posterior mean at candidate points (in objective units).
    sigma : ndarray, shape (2000,)
        GP posterior standard deviation at candidate points (in objective units).
    f_best : float
        Best observed value so far.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values are preferred.
    """
    global _obs_count
    t = _obs_count
    _obs_count += 1

    # Time-varying beta for GP-UCB (LCB for minimization)
    # Using a moderate beta to balance exploration/exploitation
    beta = 1.0 + 0.1 * t

    # LCB score for minimization: we want to maximize (beta * sigma - mu)
    # which is equivalent to minimizing (mu - beta * sigma)
    lcb_score = beta * sigma - mu

    # Standardized EI for balancing
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Normalize EI to [0, 1]
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)

    # Normalize LCB score to [0, 1]
    lcb_min = np.min(lcb_score)
    lcb_max = np.max(lcb_score)
    lcb_range = lcb_max - lcb_min
    if lcb_range > 1e-12:
        lcb_norm = (lcb_score - lcb_min) / lcb_range
    else:
        lcb_norm = np.zeros_like(lcb_score)

    # Combine: 50% LCB + 50% EI
    # LCB provides exploration (high sigma) and exploitation (low mu)
    # EI provides exploitation relative to f_best
    scores = 0.5 * lcb_norm + 0.5 * ei_norm

    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
