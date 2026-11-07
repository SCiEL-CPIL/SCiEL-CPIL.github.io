"""Acquisition function for Bayesian optimization of a 2D RBF mixture."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 325.0  # Safety margin

def _time_remaining():
    return _DEADLINE - time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Probability of Improvement (PI) for minimization.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP posterior mean at candidate points.
    sigma : ndarray, shape (2000,)
        GP posterior std at candidate points.
    f_best : float
        Best (lowest) observed objective value so far.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising points.
    """
    # Check deadline
    if _time_remaining() < 0.5:
        # Fallback: simple PI calculation
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        pi = norm.cdf(z)
        return np.nan_to_num(pi, nan=0.0, posinf=0.0, neginf=0.0)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement for minimization
    # PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    pi = norm.cdf(z)
    
    # Ensure finite output
    pi = np.nan_to_num(pi, nan=0.0, posinf=0.0, neginf=0.0)
    
    return pi
# EVOLVE-BLOCK-END
