"""Acquisition function for Bayesian optimization of Hartmann 6D."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 320s out of 330s wall-clock limit


def _check_deadline():
    """Return True if we are past the safety deadline."""
    return time.time() > _DEADLINE


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: returns 2000 finite scores.
    Higher score = more likely to be evaluated next.
    
    Uses Pure Expected Improvement (EI) for minimization.
    This approach relies on the GP posterior to balance exploration and exploitation
    naturally, avoiding heuristic exploration bonuses that can mislead the search
    in high-dimensional spaces.
    """
    global _START_TIME, _DEADLINE
    
    # Deadline guard
    if _check_deadline():
        # Fallback: exploit the most promising points (lowest mu)
        # Ensure finite values
        safe_mu = np.where(np.isfinite(mu), mu, f_best)
        return -safe_mu
    
    # Sanitize inputs
    # sigma must be positive for EI calculation
    # Clip sigma to a small positive value to avoid division by zero
    # and to handle cases where the GP is very certain (sigma ~ 0)
    sigma = np.maximum(sigma, 1e-10)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-10)
    
    # mu should be finite. If not, treat it as very bad (high value for minimization)
    # or replace with f_best to make EI 0. Replacing with f_best is safer for stability.
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Expected Improvement for minimization
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    # To prevent overflow in norm.cdf/pdf for extreme z values,
    # we can clip z. However, norm.cdf and norm.pdf in scipy are generally stable.
    # Let's compute z carefully.
    z = (f_best - mu) / sigma
    
    # Clip z to a reasonable range to avoid numerical issues in extreme tails
    # Phi(z) -> 1 for z >> 0, Phi(z) -> 0 for z << 0
    # phi(z) -> 0 for |z| >> 0
    # Clipping to [-10, 10] is safe and standard.
    z = np.clip(z, -10, 10)
    
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Ensure all scores are finite and non-negative (EI should be >= 0)
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
