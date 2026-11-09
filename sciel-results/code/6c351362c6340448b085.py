import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function.
    
    Parameters:
    mu (np.ndarray): Posterior mean of shape (2000,)
    sigma (np.ndarray): Posterior standard deviation of shape (2000,)
    f_best (float): Best observed function value so far.
    
    Returns:
    np.ndarray: Scores of shape (2000,), higher is better.
    """
    start_time = time.time()
    
    n = len(mu)
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute Expected Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: For minimization, improvement is f_best - f(x).
    # If f(x) < f_best, improvement is positive.
    # E[improvement] = E[max(0, f_best - f(x))]
    # = (f_best - mu) * P(f(x) < f_best) + sigma * E[(f_best - f(x)) | f(x) < f_best] * P(f(x) < f_best)?
    # Standard formula: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure finite values
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Check deadline
    if time.time() - start_time > 2.5:
        return ei
        
    return ei
