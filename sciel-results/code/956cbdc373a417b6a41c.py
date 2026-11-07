import numpy as np
import time

_start_time = time.time()
_DEADLINE = 320.0

def acquisition(mu, sigma, f_best):
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Fallback: exploit best observed
        return -mu.copy()
    
    # Probability of Improvement (PI) with dynamic threshold xi
    # PI(x) = P(f(x) < f_best - xi) = Phi((f_best - xi - mu(x)) / sigma(x))
    # Higher PI means higher probability of significant improvement
    
    # Dynamic threshold: xi scales with f_best to target relative improvement
    # Early on (large f_best), xi is larger -> more exploration
    # Late on (small f_best), xi is smaller -> more exploitation
    xi = 0.1 * f_best
    
    # Handle zero or very small sigma
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Standardized distance with threshold
    z = (f_best - xi - mu) / sigma_safe
    
    # Standard normal CDF
    from scipy.stats import norm
    cdf = norm.cdf(z)
    
    # PI score
    pi_score = cdf
    
    # Ensure finite
    pi_score = np.nan_to_num(pi_score, nan=0.0, posinf=1.0, neginf=0.0)
    
    return pi_score
