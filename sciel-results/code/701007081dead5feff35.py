"""Acquisition function for Bayesian optimization of Michalewicz 2D using Expected Improvement."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 320  # safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization (minimization).
    Returns scores for 2000 candidate points. Higher score = more likely to be selected.
    """
    # Deadline guard
    if time.time() - _start_time > _TIME_LIMIT:
        # Pure exploitation fallback
        return -mu
    
    n = len(mu)
    
    # Sanitize inputs
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Use Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    # Compute the normalized improvement
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -50, 50)
    
    # Compute EI
    # For minimization, we want to find points where f(x) < f_best
    # EI = E[max(f_best - f(x), 0)]
    # = (f_best - mu) * P(f(x) < f_best) + sigma * phi(z)
    # where P(f(x) < f_best) = Phi(z)
    
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Ensure finite scores
    ei[~np.isfinite(ei)] = 0.0
    
    return ei
# EVOLVE-BLOCK-END
