import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    
    # Safety margin: 300s limit. 
    # If we are close to the limit, return a simple fallback.
    # Vectorized ops are fast, so this is mostly a precaution.
    
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    # Probability of Improvement (PI)
    # For minimization:
    # P(f(x) < f_best) = Phi( (f_best - mu) / sigma )
    # We want higher score for higher probability.
    
    improvement = f_best - mu
    z = improvement / sigma_safe
    
    # Calculate PI
    pi = norm.cdf(z)
    
    # Ensure finite
    pi = np.where(np.isfinite(pi), pi, 0.0)
    
    # Deadline guard check
    elapsed = time.time() - start_time
    if elapsed > 290:
        # Return the computed pi which is already done.
        pass
        
    return pi
