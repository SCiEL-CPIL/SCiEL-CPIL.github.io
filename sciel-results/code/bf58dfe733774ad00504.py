import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Epsilon to prevent division by zero
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues in norm.cdf/pdf for very large/small values
    # Although scipy handles it reasonably, clipping is safe.
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    scores = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check deadline
    if time.time() - start_time > 320:
         pass
         
    return scores
