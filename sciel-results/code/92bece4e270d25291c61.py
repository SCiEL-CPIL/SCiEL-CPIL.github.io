"""Acquisition for Bayesian optimization of Michalewicz 2D using Expected Improvement."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_DEADLINE = time.time() + 290.0  # safety margin before 300s kill

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement acquisition function for Bayesian optimization (minimization).
    Uses adaptive exploration parameter xi that scales with posterior uncertainty.
    Higher score = more likely to be selected.
    """
    # Deadline guard: fall back to pure exploitation if running out of time
    if time.time() > _DEADLINE:
        return -mu
    
    # Sanitize inputs
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Adaptive exploration parameter: scales with mean posterior uncertainty
    # This helps balance exploitation and exploration dynamically
    mean_sigma = np.mean(sigma_safe)
    xi = 0.1 * f_best * (mean_sigma / (np.abs(f_best) + 1e-10))
    
    # Compute Expected Improvement for minimization
    # EI = E[max(f_best - f(x), 0)]
    # For f(x) ~ N(mu, sigma^2):
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # With xi offset: use (f_best - xi) instead of f_best
    
    effective_best = f_best - xi
    
    # Standardized distance
    z = (effective_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues with extreme values
    z = np.clip(z, -10, 10)
    
    # Expected Improvement formula
    ei = (effective_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # EI is always non-negative; higher means more improvement expected
    scores = ei
    
    # Ensure finite scores
    scores[~np.isfinite(scores)] = 0.0
    
    return scores
# EVOLVE-BLOCK-END
