"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() - _start_time > 320:
        return np.zeros_like(mu)

    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Calculate EI
    # Note: If sigma is very small and mu < f_best, z is large positive, Phi(z)~1, phi(z)~0, EI ~ f_best - mu
    # If sigma is very small and mu > f_best, z is large negative, Phi(z)~0, phi(z)~0, EI ~ 0
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI / Cost
    # This is a structural change from PI/Cost^gamma to EI/Cost
    # EI provides a stronger signal for exploitation than PI, potentially counteracting
    # the bias towards cheap regions if the cost penalty is linear.
    scores = ei / (cost + 1e-9)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
