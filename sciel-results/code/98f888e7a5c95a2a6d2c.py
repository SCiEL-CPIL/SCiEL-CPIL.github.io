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

    # Time-varying beta for GP-UCB
    beta = np.sqrt(2.0 * t * np.log(1.0 + t))

    # GP-UCB score: mu + sqrt(beta) * sigma
    ucb_score = mu + np.sqrt(beta) * sigma

    # Standardized EI for balancing
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Normalize EI to be comparable in scale with UCB
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)

    # Normalize UCB to [0, 1] range for combination
    ucb_min = np.min(ucb_score)
    ucb_max = np.max(ucb_score)
    ucb_range = ucb_max - ucb_min
    if ucb_range > 1e-12:
        ucb_norm = (ucb_score - ucb_min) / ucb_range
    else:
        ucb_norm = np.zeros_like(ucb_score)

    # Combine: 70% UCB + 30% normalized EI
    scores = 0.7 * ucb_norm + 0.3 * ei_norm

    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
