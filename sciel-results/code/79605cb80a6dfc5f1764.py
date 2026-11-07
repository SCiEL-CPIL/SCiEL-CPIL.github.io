"""Bayesian optimization acquisition for Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_obs_count = 5
_start_time = time.time()
_DEADLINE = 325.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best):
    """
    Combined GP-LCB (for minimization) and normalized Expected Improvement acquisition function.
    
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

    # Deadline guard: if we're close to the limit, heavily favor exploitation
    elapsed = time.time() - _start_time
    time_critical = (elapsed > _DEADLINE)

    # Time-varying beta for GP-LCB (theoretically grounded exploration)
    beta = np.sqrt(2.0 * t * np.log(1.0 + t))
    sqrt_beta = np.sqrt(beta)

    # For minimization: LCB = mu - sqrt_beta * sigma
    # We want to MAXIMIZE the LCB score (higher = more promising)
    lcb_score = mu - sqrt_beta * sigma

    # Expected Improvement for minimization
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    if time_critical:
        # Near deadline: pure exploitation via EI only
        ei_max = np.max(ei)
        if ei_max > 1e-12:
            scores = ei / ei_max
        else:
            scores = np.zeros_like(ei)
    else:
        # Normalize EI to [0, 1]
        ei_max = np.max(ei)
        if ei_max > 1e-12:
            ei_norm = ei / ei_max
        else:
            ei_norm = np.zeros_like(ei)

        # Normalize LCB to [0, 1] (higher LCB is better for minimization)
        lcb_min = np.min(lcb_score)
        lcb_max = np.max(lcb_score)
        lcb_range = lcb_max - lcb_min
        if lcb_range > 1e-12:
            lcb_norm = (lcb_score - lcb_min) / lcb_range
        else:
            lcb_norm = np.zeros_like(lcb_score)

        # Combine: 70% LCB + 30% normalized EI
        scores = 0.7 * lcb_norm + 0.3 * ei_norm

    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
