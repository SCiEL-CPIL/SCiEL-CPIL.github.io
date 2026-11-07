"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0  # wall-clock seconds total allowed (5s safety margin)

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Styblinski-Tang 1D.
    
    Uses Squared Expected Improvement (SEI) to emphasize large improvements
    and robustly handle the multi-modal nature of the function.
    
    Args:
        mu: array of shape (2000,) with posterior means
        sigma: array of shape (2000,) with posterior standard deviations
        f_best: lowest observed function value
    
    Returns:
        array of shape (2000,) with acquisition scores (higher is better)
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Return a simple fallback score: exploit best mean
        return -mu.copy()
    
    # Guard against negative or zero sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Standard Expected Improvement (minimization)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Squared Expected Improvement
    # This emphasizes candidates with high expected improvement more strongly,
    # which can help in escaping local minima by focusing on regions with
    # significant potential for improvement.
    sei = ei ** 2
    
    # Ensure finite output
    sei = np.where(np.isfinite(sei), sei, 0.0)
    
    return sei
# EVOLVE-BLOCK-END
