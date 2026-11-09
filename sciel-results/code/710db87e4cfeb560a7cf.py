"""Rewritten program using Pure Expected Improvement for Bayesian optimization acquisition."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Track global start time for deadline guard
_global_start_time = None
_TIMEOUT_MARGIN = 5.0  # seconds to spare before 330s limit
_TOTAL_LIMIT = 330.0


def _check_deadline():
    """Check if we are approaching the timeout limit."""
    global _global_start_time
    if _global_start_time is None:
        _global_start_time = time.time()
    elapsed = time.time() - _global_start_time
    return elapsed > (_TOTAL_LIMIT - _TIMEOUT_MARGIN)


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    
    Args:
        mu: array of shape (2000,) - posterior mean
        sigma: array of shape (2000,) - posterior standard deviation
        f_best: float - lowest observed value
    
    Returns:
        array of shape (2000,) - scores, higher means evaluate next
    """
    global _global_start_time
    if _global_start_time is None:
        _global_start_time = time.time()
    
    # Check deadline - if close to timeout, return a safe default
    if _check_deadline():
        # Return Expected Improvement as a safe fallback
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        return (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure sigma is positive to avoid division by zero
    # A small epsilon is added to sigma to encourage exploration in areas with near-zero uncertainty
    # if mu is slightly better than f_best, or to stabilize the calculation.
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI(x) = E[max(f_best - f(x), 0)]
    # where f(x) ~ N(mu(x), sigma(x)^2)
    
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure all scores are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
