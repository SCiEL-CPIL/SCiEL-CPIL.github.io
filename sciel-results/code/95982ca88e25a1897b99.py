import numpy as np
from scipy.stats import norm
import time

_start_time = None

def acquisition(mu, sigma, f_best):
    global _start_time
    
    # Initialize start time on first call
    if _start_time is None:
        _start_time = time.time()
    
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-8)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-8)
    
    # Compute Expected Improvement for minimization
    # EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x))/sigma(x)) + sigma(x) * phi((f_best - mu(x))/sigma(x))
    improvement = f_best - mu
    
    # z-score
    z = improvement / sigma_safe
    
    # Standard normal CDF and PDF
    phi = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Expected Improvement
    ei = improvement * phi + sigma_safe * pdf
    
    # Ensure all scores are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
