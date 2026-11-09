"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_START_TIME = time.time()
_TIMEOUT = 290  # Safety margin before 300s


def acquisition(mu, sigma, f_best):
    """
    Acquisition function that blends Expected Improvement (EI) with Upper Confidence Bound (UCB).
    The blend adapts based on how much uncertainty remains, favoring exploration early and
    exploitation later.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Posterior mean of the GP at candidate points.
    sigma : array-like, shape (2000,)
        Posterior standard deviation of the GP at candidate points.
    f_best : float
        Best observed function value so far (minimum).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    # Check deadline - if we're running out of time, return a fast fallback
    if time.time() - _START_TIME > _TIMEOUT:
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return ei

    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)

    # Compute Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Compute UCB: mu - beta * sigma (for minimization, we want low mu and high uncertainty)
    # beta controls the exploration-exploitation tradeoff
    # A common choice is beta = 1.0 to 2.0, but we can adapt it

    # Adaptive beta: increase exploration when sigma is large relative to the range of f_best
    # Estimate scale from mu and sigma
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    mu_range = max(mu_max - mu_min, 1e-9)
    sigma_mean = np.mean(sigma_safe)
    
    # If sigma is large relative to the spread of mu, we're in early exploration
    # Use a higher beta for more exploration
    exploration_factor = np.clip(sigma_mean / (mu_range + 1e-9), 0.5, 2.0)
    beta = 1.0 + exploration_factor  # beta between 1.5 and 3.0

    # UCB for minimization: we want to minimize, so the "upper confidence bound" 
    # in the sense of "best case" is mu - beta * sigma (lower is better)
    # But for acquisition, we want higher score = more promising
    # So we negate: score_ucb = -(mu - beta * sigma) = -mu + beta * sigma
    ucb = -mu + beta * sigma_safe

    # Normalize EI and UCB to comparable scales
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = max(ei_max - ei_min, 1e-9)
    ei_norm = (ei - ei_min) / ei_range

    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = max(ucb_max - ucb_min, 1e-9)
    ucb_norm = (ucb - ucb_min) / ucb_range

    # Blend: weight EI more when we have good estimates (low sigma), UCB more when uncertain
    # Use the relative uncertainty to determine the blend
    # When sigma is small, trust EI more; when sigma is large, trust UCB more
    sigma_weight = np.clip(sigma_safe / (sigma_mean + 1e-9), 0.0, 1.0)
    # Blend coefficient: 0.5 is balanced, higher favors UCB (exploration)
    blend_alpha = 0.3 + 0.4 * sigma_weight  # Between 0.3 and 0.7

    scores = blend_alpha * ei_norm + (1 - blend_alpha) * ucb_norm

    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
