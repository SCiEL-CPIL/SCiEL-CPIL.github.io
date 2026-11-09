"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard
_DEADLINE = 300.0
_start_time = time.time()

def _check_deadline():
    return (time.time() - _start_time) > _DEADLINE

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    
    EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    where z = (f_best - mu(x)) / sigma(x)
    
    We return the EI scores. Higher is better.
    """
    if _check_deadline():
        # Fallback: simple UCB
        return -mu + 2.0 * sigma

    # Ensure sigma is strictly positive to avoid division by zero
    # The problem states sigma is a standard deviation, so it should be > 0.
    # But numerical issues might cause 0.
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Calculate z-score
    # z = (f_best - mu) / sigma
    with np.errstate(divide='ignore', invalid='ignore'):
        z = (f_best - mu) / sigma_safe
        
        # Calculate EI
        # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
        phi_z = norm.pdf(z)
        Phi_z = norm.cdf(z)
        
        ei_scores = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Handle any numerical nan/inf
    ei_scores = np.where(np.isfinite(ei_scores), ei_scores, 0.0)
    
    return ei_scores
# EVOLVE-BLOCK-END
