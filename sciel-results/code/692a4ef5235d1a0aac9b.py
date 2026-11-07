"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Use Expected Improvement (EI) for minimization.
    # EI balances exploration and exploitation by maximizing the expected
    # reduction in the best observed value.
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Calculate z-score: (f_best - mu) / sigma
    # If mu is much smaller than f_best, z is large positive.
    # If mu is much larger than f_best, z is large negative.
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement
    # E[I] = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the CDF and phi is the PDF of the standard normal distribution.
    
    # Use scipy.stats.norm for numerical stability
    phi_z = norm.pdf(z)
    phi_z_cdf = norm.cdf(z)
    
    ei = (f_best - mu) * phi_z_cdf + sigma_safe * phi_z
    
    # Handle numerical issues where sigma is very small
    # If sigma is effectively 0, EI should be max(0, f_best - mu)
    # The formula above converges to this, but let's ensure no NaNs or Infs.
    
    # Replace non-finite values with -inf (or 0 if we want to avoid negative scores? 
    # The task says "higher meaning evaluate next". 
    # EI is always non-negative. 
    # If a point is known to be worse than f_best, EI is 0.
    # We should probably return the EI value directly.
    
    # Ensure finite output
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
