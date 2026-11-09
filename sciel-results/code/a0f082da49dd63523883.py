import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    
    # Safety margin: 300s limit. We check every call.
    # If we are close to the limit, we might want to return a simple fallback.
    # But vectorized ops are fast.
    
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    # Exploration parameter
    xi = 1e-2
    
    improvement = f_best - mu - xi
    z = improvement / sigma_safe
    
    # Calculate EI
    # Use scipy.stats.norm for CDF and PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = improvement * cdf_z + sigma_safe * pdf_z
    
    # Ensure finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Deadline guard check
    elapsed = time.time() - start_time
    if elapsed > 290:
        # This should never happen for 2000 points, but if it does,
        # we return the computed ei which is already done.
        # If the computation itself was hanging, we can't fix it mid-way.
        pass
        
    return ei
