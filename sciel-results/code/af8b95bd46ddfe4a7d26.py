"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import ndtr

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # Safety margin before 300s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d.
    
    Uses Expected Hypar (EHypR): E[max(f_best, f(x)) / f(x)]
    where f(x) ~ N(mu, sigma^2).
    
    For minimization, we want to maximize the probability that f(x) < f_best
    weighted by how much better it is. EHypR directly optimizes simple regret.
    
    score = E[max(f_best, f(x)) / f(x)]
    = f_best * P(f(x) > f_best) + E[f(x)/f(x)] * P(f(x) <= f_best)
    = f_best * P(f(x) > f_best) + 1 * P(f(x) <= f_best)
    
    Wait, that simplifies to:
    = f_best * (1 - Phi(z)) + Phi(z)
    where z = (f_best - mu) / sigma
    
    So score = Phi(z) + f_best * (1 - Phi(z))
    
    This is a convex combination of 1 (improvement) and f_best (no improvement).
    Higher score means better expected ratio.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: simple EI
        sigma_safe = np.maximum(sigma, 1e-12)
        z = (f_best - mu) / sigma_safe
        phi_z = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
        ei = (f_best - mu) * ndtr(z) + sigma_safe * phi_z
        return np.where(np.isfinite(ei), ei, 0.0)

    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # EHypR score = Phi(z) + f_best * (1 - Phi(z))
    # = 1 + (f_best - 1) * (1 - Phi(z))
    # = 1 + (f_best - 1) * ndtr(-z)
    phi_z = ndtr(z)
    score = phi_z + f_best * (1.0 - phi_z)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
