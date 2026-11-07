import numpy as np
import time
from scipy.stats import norm

_T0 = None
_DEADLINE = None

def _ensure_deadline():
    global _T0, _DEADLINE
    if _T0 is None:
        _T0 = time.perf_counter()
        _DEADLINE = _T0 + 295.0  # 5s margin for safety
    return _DEADLINE

def acquisition(mu, sigma, f_best):
    deadline = _ensure_deadline()
    elapsed = time.perf_counter() - _T0
    
    # Fallback if time is running out
    if elapsed > _DEADLINE - 5.0:
        # Simple LCB fallback
        return -mu + 1.0 * sigma

    # Expected Improvement (EI)
    # We want to maximize the expected improvement over f_best
    # Improvement I(x) = max(f_best - f(x), 0)
    # E[I(x)] = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    # Handle sigma == 0 to avoid division by zero
    # If sigma is 0, EI = max(f_best - mu, 0)
    sigma_safe = np.where(sigma > 1e-12, sigma, 1e-12)
    
    Z = (f_best - mu) / sigma_safe
    
    # Phi(Z) is the CDF of the standard normal distribution
    # phi(Z) is the PDF of the standard normal distribution
    phi_z = norm.pdf(Z)
    Phi_z = norm.cdf(Z)
    
    score = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
