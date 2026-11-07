# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 315.0  # Safety margin before 330s

def _check_time():
    return (time.time() - _START_TIME) > _TIME_LIMIT

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 3d.
    Returns 2000 scores, higher means evaluate next.
    """
    # Ensure we have the right shape and type
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: if sigma has zeros or negatives, clip them
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Check time budget
    if _check_time():
        # Fallback to simple EI if we're running out of time
        z = (f_best - mu) / sigma_safe
        cdf = norm.cdf(z)
        pdf = norm.pdf(z)
        scores = (f_best - mu) * cdf + sigma_safe * pdf
        return np.where(np.isfinite(scores), scores, -np.inf)
    
    # Strategy: Use a robust Expected Improvement (EI) acquisition function.
    # EI is known to be very effective for finding the global minimum in multi-modal functions
    # like Hartmann 3D, especially when the GP is well-calibrated.
    # The baseline score of 1.0 suggests that a pure or near-pure EI strategy is strong.
    # Previous attempts at hybrids or REI scored lower, implying that the specific
    # scaling or combination of terms was detrimental.
    
    # We will use standard EI:
    # EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    # where z = (f_best - mu(x)) / sigma(x)
    # Phi is the standard normal CDF and phi is the standard normal PDF.
    
    # This is the gold standard for single-point Bayesian optimization.
    # It balances exploitation (high f_best - mu) and exploration (high sigma)
    # in a theoretically optimal way for the expected improvement.
    
    # Compute z
    z = (f_best - mu) / sigma_safe
    
    # Compute CDF and PDF
    # Using scipy.stats.norm for accuracy
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Compute EI
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Ensure all scores are finite
    # If any value is not finite, replace with -inf to ensure it's not selected
    ei = np.where(np.isfinite(ei), ei, -np.inf)
    
    return ei

# EVOLVE-BLOCK-END
