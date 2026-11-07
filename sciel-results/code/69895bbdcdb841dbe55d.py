"""Bayesian optimization acquisition for Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_obs_count = 5
_start_time = time.time()
_TIMEOUT = 315.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement acquisition function for minimization.
    
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

    # Check time limit: if we are close to the deadline, return quickly
    elapsed = time.time() - _start_time
    if elapsed > _TIMEOUT:
        # Fallback: just return -mu (pure exploitation) to finish quickly
        return -mu

    # Expected Improvement for minimization
    # EI(x) = E[max(f_best - f(x), 0)]
    # For GP: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure finite values
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
