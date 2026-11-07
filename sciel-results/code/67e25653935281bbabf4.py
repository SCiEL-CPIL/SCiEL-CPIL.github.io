"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: start time for the overall acquisition call
_acq_start_time = None
_DEADLINE_SECONDS = 290.0  # Safety margin before 300s limit

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) for minimization with adaptive exploration.
    
    EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu) / sigma
    
    We use EI as the primary score since it balances exploitation and exploration
    well for multimodal functions like Branin.
    """
    global _acq_start_time
    if _acq_start_time is None:
        _acq_start_time = time.time()
    
    elapsed = time.time() - _acq_start_time
    if elapsed > _DEADLINE_SECONDS:
        # Return a safe default: prefer points with low mean
        return -mu.copy()
    
    # Compute EI for minimization
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei_scores = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure all scores are finite
    scores = np.nan_to_num(ei_scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
