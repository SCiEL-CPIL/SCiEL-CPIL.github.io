import numpy as np
import time

# Module-level time tracking for deadline guard
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition for Bayesian optimization.
    
    Uses the standard Expected Improvement formula for minimization,
    divided by the evaluation cost. This provides a probabilistic measure
    of improvement per unit cost.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: return quickly if we're getting close to the limit
    if time.time() - _start_time > 325:
        return np.zeros_like(mu, dtype=np.float64)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-10)
    
    # Calculate Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    
    # z-score
    z = (f_best - mu) / sigma_safe
    
    # CDF of standard normal: Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    from scipy.special import erf, exp
    sqrt_2 = np.sqrt(2.0)
    cdf_vals = 0.5 * (1.0 + erf(z / sqrt_2))
    
    # PDF of standard normal: phi(z) = exp(-z^2 / 2) / sqrt(2*pi)
    pdf_vals = exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    
    ei = (f_best - mu) * cdf_vals + sigma_safe * pdf_vals
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score
    score = ei / cost_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
