"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of a 2D RBF mixture.
    
    Uses Expected Improvement (EI) for minimization with numerical safeguards.
    
    Parameters:
        mu: array of shape (2000,), posterior mean at candidates
        sigma: array of shape (2000,), posterior std at candidates
        f_best: scalar, best (lowest) observed value so far
    
    Returns:
        array of shape (2000,), scores (higher = more promising to evaluate next)
    """
    # Ensure sigma is positive and finite to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement: (f_best - mu) / sigma
    # For minimization, we want mu < f_best, so (f_best - mu) > 0 is good
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues with extreme values
    # norm.cdf and norm.pdf handle large values, but clipping adds safety
    z_clipped = np.clip(z, -10.0, 10.0)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Ensure all values are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
