# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import erfc

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using numerically stable Expected Improvement (EI).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety checks
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-8)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Calculate z-score
    # z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues in erf/erfc for very large/small values
    z = (f_best - mu) / sigma
    z = np.clip(z, -50, 50)
    
    sqrt2 = np.sqrt(2.0)
    
    # Phi(z) = 0.5 * erfc(-z / sqrt2)
    Phi = 0.5 * erfc(-z / sqrt2)
    
    # phi(z) = (1/sqrt(2*pi)) * exp(-z^2/2)
    # Use log-space for stability if z is large, but z is clipped to 50, so exp(-1250) is 0.
    # phi(50) is extremely small.
    phi = np.exp(-0.5 * z * z - 0.5 * np.log(2.0 * np.pi))
    
    # Expected Improvement
    # For minimization, improvement is f_best - f(x)
    ei = (f_best - mu) * Phi + sigma * phi
    
    # Ensure non-negative (EI should theoretically be >= 0)
    ei = np.maximum(ei, 0.0)
    
    # Ensure finite output
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
