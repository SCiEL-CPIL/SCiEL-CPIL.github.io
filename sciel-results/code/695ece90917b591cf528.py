import numpy as np
import time
from scipy.stats import norm

_start_time = None

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def acquisition(mu, sigma, f_best):
    # Deadline guard
    elapsed = _get_elapsed()
    if elapsed > 325:
        # Return a simple score quickly
        mu = np.asarray(mu)
        scores = -mu
        scores = np.where(np.isfinite(scores), scores, -np.inf)
        return scores
    
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # Ensure sigma is positive and finite
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-12)
    
    # Knowledge Gradient approximation for minimization
    # KG(x) = E[max(f_best - f(x), 0)]
    # where f(x) ~ N(mu(x), sigma(x)^2)
    # This is the Expected Improvement formula:
    # EI(x) = sigma(x) * [z * Phi(z) + phi(z)] where z = (f_best - mu(x)) / sigma(x)
    
    z = (f_best - mu) / sigma_safe
    
    # Compute EI = sigma * (z * Phi(z) + phi(z))
    # Using scipy's norm for numerical stability
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    scores = sigma_safe * (z * Phi_z + phi_z)
    
    # Handle non-finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
