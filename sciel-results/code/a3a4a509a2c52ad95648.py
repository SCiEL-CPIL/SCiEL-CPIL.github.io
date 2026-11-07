import numpy as np
import time
from scipy.stats import norm

# Global timer for deadline guard
_start_time = time.time()
_TIME_LIMIT = 315.0  # Safety margin from 330s

def acquisition(mu, sigma, f_best):
    # Check time
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        # If we are running out of time, return a safe fallback
        return np.full(len(mu), -1e10)

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Handle edge cases
    if mu.size == 0:
        return np.array([])

    # Expected Improvement (EI) for minimization
    # EI(x) = E[max(f_best - f(x), 0)]
    # For f(x) ~ N(mu, sigma^2):
    # EI(x) = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # where Z = (f_best - mu) / sigma
    
    # Add a small exploration parameter xi to encourage visiting
    # areas that are not strictly better than the current best,
    # which helps escape local minima in multi-modal functions.
    xi = 0.01
    
    # Ensure sigma is positive to avoid division by zero
    # Using a very small epsilon
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate Z
    z = (f_best + xi - mu) / sigma_safe
    
    # Calculate EI
    # norm.cdf and norm.pdf are vectorized
    ei = (f_best + xi - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure finiteness
    ei = np.where(np.isfinite(ei), ei, -1e10)
    
    return ei
