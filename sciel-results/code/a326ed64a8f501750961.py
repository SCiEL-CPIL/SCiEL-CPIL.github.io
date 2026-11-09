import numpy as np
from scipy.special import erf
import time

def acquisition(mu, sigma, f_best):
    # Deadline guard (safety margin)
    # Note: acquisition is called frequently, so we keep this check lightweight.
    # If this function were part of a larger loop, we would track start time.
    # Here, we assume the caller handles the main loop timeout, but we ensure
    # this function itself is robust and fast.
    
    # Ensure sigma is not zero to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-6)
    
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid extreme values where erf might lose precision
    # (though erf is very stable, clipping is good practice)
    z_clipped = np.clip(z, -10, 10)
    
    # Use scipy.special.erf for CDF of standard normal
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    cdf = 0.5 * (1.0 + erf(z_clipped / np.sqrt(2.0)))
    
    # PDF of standard normal
    # phi(z) = 1/sqrt(2*pi) * exp(-z^2 / 2)
    pdf = np.exp(-0.5 * z_clipped * z_clipped) / np.sqrt(2.0 * np.pi)
    
    # Standard EI
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Ensure non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    return ei
