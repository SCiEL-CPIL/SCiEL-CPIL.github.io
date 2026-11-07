# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d on [-5, 5].
    Uses Expected Improvement (EI), which is a robust and theoretically grounded
    method for balancing exploration and exploitation in Bayesian optimization.
    """
    start_time = time.time()
    deadline = 320.0  # Safety margin before 330s limit

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Safety checks
    # Ensure sigma is positive and finite
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    sigma = np.maximum(sigma, 1e-8)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, f_best)

    # Compute z-score for EI
    # z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues in cdf/pdf for extreme values
    # Although scipy handles inf, clipping is safer for stability
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)

    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    ei = (f_best - mu) * cdf + sigma * pdf
    
    # EI is always non-negative by definition in the formula, 
    # but due to floating point errors, it might be slightly negative.
    ei = np.maximum(ei, 0.0)
    
    # Handle any non-finite values
    ei = np.where(np.isfinite(ei), ei, 0.0)

    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Fallback: return a simple exploitation score (negative mean)
        # This ensures we return *something* valid
        return -mu
    
    return ei
# EVOLVE-BLOCK-END
