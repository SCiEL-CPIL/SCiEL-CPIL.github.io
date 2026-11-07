"""Bayesian optimization acquisition for Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_obs_count = 5
_start_time = time.time()
_TIMEOUT = 320.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best):
    """
    Combined GP-UCB (LCB for minimization) and normalized Expected Improvement acquisition function.
    
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

    # Check time limit: if we are close to the deadline, prioritize exploitation (current best)
    # Since we don't know coordinates, we can't directly return the best point.
    # However, we can bias towards points with very low mu (exploitation).
    # If time is critical, we might want to stop exploring.
    # For now, we just ensure we don't hang. The operations are fast.
    
    # Time-varying beta for GP-UCB
    # Dampen the growth slightly to avoid over-exploration in later stages if needed,
    # but the directive asks for the theoretical form.
    beta = np.sqrt(2.0 * t * np.log(1.0 + t))
    
    # Scale beta down if t is large? 
    # For Beale, the function is smooth. 
    # Let's stick to the theoretical beta but ensure sign is correct for minimization.
    
    # For minimization, we want to maximize (-mu + beta * sigma)
    # This is equivalent to minimizing (mu - beta * sigma)
    ucb_score = -mu + beta * sigma

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
    # If time is running out, we might want to shift weight to exploitation (EI)
    # EI is 0 if no improvement is expected. 
    # Actually, if time is out, we should probably just return the point with lowest mu?
    # But we can't change the structure drastically.
    # Let's just use the standard combination.
    
    scores = 0.7 * ucb_norm + 0.3 * ei_norm

    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
