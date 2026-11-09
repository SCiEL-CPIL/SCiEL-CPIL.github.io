import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best):
        elapsed = time.time() - _start_time
        if elapsed > _DEADLINE:
            return -mu # Fallback to exploitation
        
        sigma_safe = np.maximum(sigma, 1e-9)
        
        # Dynamic xi: encourage significant improvement
        # Use a fraction of the mean uncertainty
        mean_sigma = np.mean(sigma_safe)
        xi = 0.05 * mean_sigma
        
        # Standardized improvement
        z = (f_best - xi - mu) / sigma_safe
        z_clipped = np.clip(z, -10.0, 10.0)
        
        # EI with xi
        ei = (f_best - xi - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
        
        # Ensure finite
        ei = np.where(np.isfinite(ei), ei, 0.0)
        
        return ei
