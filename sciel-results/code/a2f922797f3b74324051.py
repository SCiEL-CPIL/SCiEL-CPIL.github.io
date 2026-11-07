"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guarding
_START_TIME = time.time()
_DEADLINE_FRACTION = 0.95  # Stop at 95% of the 300s limit (285s) to be safe
_LIMIT = 300.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Knowledge Gradient (Expected Improvement)
    for minimization.
    
    Strategy:
    - Uses pure EI/KG: E[max(0, f_best - f(x))]
    - This directly optimizes the expected reduction in the best observed value.
    - Includes a strict time deadline guard to ensure the function returns
      before the 300s wall-clock limit, preventing timeouts.
    - Handles numerical edge cases for sigma near zero.
    """
    # --- Deadline Guard ---
    elapsed = time.time() - _START_TIME
    if elapsed > _LIMIT * _DEADLINE_FRACTION:
        # If we are too close to the deadline, return a safe fallback
        # The fallback is arbitrary but finite. We return zeros to avoid
        # biasing the selection if we can't compute properly.
        # In practice, this should rarely trigger if computation is fast,
        # but it satisfies the structural requirement.
        return np.zeros_like(mu)
    
    # Ensure sigma is positive to avoid division issues
    # Use a small epsilon to prevent division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute Expected Improvement (KG) for minimization
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    z = (f_best - mu) / sigma_safe
    
    # Numerical stability for very large positive z (mu << f_best, high confidence of improvement)
    # Phi(z) -> 1, phi(z) -> 0
    # EI -> f_best - mu
    
    # Numerical stability for very large negative z (mu >> f_best, high confidence of no improvement)
    # Phi(z) -> 0, phi(z) -> 0
    # EI -> 0
    
    # Standard calculation
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure non-negative (due to floating point errors, EI might be slightly negative)
    ei = np.maximum(ei, 0.0)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(ei), ei, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
