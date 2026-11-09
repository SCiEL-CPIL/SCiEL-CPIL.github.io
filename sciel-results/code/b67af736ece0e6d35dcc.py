"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 320.0  # 10s margin before 330s limit

def acquisition(mu, sigma, f_best):
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: return best observed (greedy) or zeros
        # Greedy is safe
        return -mu

    # Expected Improvement (EI) for minimization
    # EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    # where z = (f_best - mu(x)) / sigma(x)
    
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    
    # Compute Phi(z) and phi(z)
    Phi = norm.cdf(z)
    phi = norm.pdf(z)
    
    # EI calculation
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # Ensure finite
    if not np.all(np.isfinite(ei)):
        ei = np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=-1e6)
        
    return ei
# EVOLVE-BLOCK-END
