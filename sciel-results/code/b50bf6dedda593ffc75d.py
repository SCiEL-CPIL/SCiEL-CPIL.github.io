import numpy as np
from scipy.stats import norm

# Global start time for deadline guard, initialized on first call
_start_time = None

def acquisition(mu, sigma, f_best):
    global _start_time
    
    # Initialize start time on first call to track elapsed time
    if _start_time is None:
        import time
        _start_time = time.time()
    
    import time
    current_time = time.time()
    
    # Safety margin: if we are dangerously close to the 330s limit,
    # return a deterministic score that might not be optimal but is safe.
    # However, since we don't know the exact limit of the evaluator's internal clock,
    # we assume the function is fast enough. 
    # If we want to be strictly compliant with "return best result found with safety margin",
    # we can't do much inside a stateless acquisition function except be fast.
    # We will proceed with the calculation.
    
    # Ensure sigma is positive to avoid numerical issues
    sigma = np.asarray(sigma, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    
    # Replace any non-finite values with safe defaults
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Ensure sigma is strictly positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate Expected Improvement (EI) for minimization
    # Improvement is defined as f_best - y (since lower is better)
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # where Z = (f_best - mu) / sigma
    
    improvement = f_best - mu
    z = improvement / sigma
    
    # Compute Phi(z) and phi(z)
    # Phi is the CDF of the standard normal distribution
    # phi is the PDF of the standard normal distribution
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    ei = improvement * cdf + sigma * pdf
    
    # Ensure scores are finite
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    return ei
