import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
TIME_LIMIT = 325.0  # Safety margin before 330s kill

def _elapsed():
    return time.time() - _START_TIME

def _check_time():
    return _elapsed() > TIME_LIMIT

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    Uses Expected Improvement (EI) for minimization.
    
    mu: array of shape (2000,) - GP mean predictions
    sigma: array of shape (2000,) - GP standard deviation predictions
    f_best: float - best observed value so far
    
    Returns: array of shape (2000,) - acquisition scores (higher is better)
    """
    # Deadline guard: if we're close to the time limit, return a simple score
    if _check_time():
        # Simple exploitation score
        scores = -mu
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Expected Improvement for minimization
    # EI(x) = E[max(f_best - f(x), 0)]
    # where f(x) ~ N(mu(x), sigma(x)^2)
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    
    # Avoid division by zero
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues with extreme values
    z_clipped = np.clip(z, -10, 10)
    
    # Phi and phi are the CDF and PDF of the standard normal
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Where sigma is very small, EI should be close to max(f_best - mu, 0)
    # This is already handled by the formula above when sigma -> 0
    
    # Ensure output is finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Ensure non-negative (EI should be >= 0)
    ei = np.maximum(ei, 0.0)
    
    return ei
