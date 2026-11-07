"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global state to track time for safety
_start_time = time.time()
_DEADLINE = 300.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function.
    
    Computes the expected improvement over the best observed value f_best.
    For minimization, we define improvement as f_best - f(x) (positive when f(x) < f_best).
    
    EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu) / sigma
    
    This is a deterministic, well-established acquisition function that balances
    exploration and exploitation through the GP posterior uncertainty.
    """
    # Check time limit
    if time.time() > _DEADLINE - 10:
        # Fallback to simple exploitation if running out of time
        return -(mu - f_best)  # Equivalent to (f_best - mu)

    # Compute z-score
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma_safe
    
    # Compute EI using standard normal CDF and PDF
    # norm.cdf(z) = Phi(z), norm.pdf(z) = phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure finite scores
    if not np.all(np.isfinite(ei)):
        # Fallback: use a simple LCB-like score
        z_clipped = np.clip(z, -10, 10)
        ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    return ei
# EVOLVE-BLOCK-END
