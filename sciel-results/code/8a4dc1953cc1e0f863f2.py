import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    t_start = time.time()
    deadline = t_start + 295
    
    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Adaptive xi based on median sigma
    # If sigma is high, we are uncertain -> explore more (larger xi)
    # If sigma is low, we are certain -> exploit more (smaller xi)
    # lambda controls the strength of exploration
    # A value of 0.5-1.0 is typical
    med_sigma = np.median(sigma_safe)
    xi = 0.5 * med_sigma
    
    # Expected Improvement with xi
    improvement = f_best - mu - xi
    z = improvement / sigma_safe
    
    # EI = improvement * Phi(z) + sigma * phi(z)
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure non-negative
    ei = np.maximum(ei, 0.0)
    
    # Time guard
    if time.time() > deadline:
        return ei
        
    return ei
