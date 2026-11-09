"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START = time.time()
_DEADLINE = _START + 325.0


def acquisition(mu, sigma, f_best):
    """Log-Expected Improvement for minimization."""
    # Deadline guard: if we're running out of time, return a safe fallback
    if time.time() > _DEADLINE:
        scores = f_best - mu
        return np.where(np.isfinite(scores), scores, 0.0)

    sigma_safe = np.maximum(sigma, 1e-12)
    delta = f_best - mu  # positive when mu < f_best (improvement possible)
    
    z = delta / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    
    # Standard EI calculation
    ei = delta * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Ensure non-negative and finite
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Log-EI: transform EI to Log-EI
    # log(EI + epsilon) encourages exploration in regions with low EI but high uncertainty
    # This prevents the acquisition function from becoming too greedy (pure exploitation)
    # which is a common issue with standard EI when the noise is very low (1e-6).
    scores = np.log(ei + 1e-12)
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
