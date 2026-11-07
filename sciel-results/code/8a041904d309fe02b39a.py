"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Track start time for potential timeout guards
_start_time = time.time()
_DEADLINE = _start_time + 290  # Keep a 10s safety margin before the 300s limit

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI).
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed objective value so far.
        
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    # Ensure sigma is strictly positive to avoid division by zero in EI calculation
    # Add a small epsilon to handle cases where sigma is exactly 0
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Calculate Improvement: I = f_best - f(x)
    # We want to maximize the expected value of I, but only where I > 0
    # Standard EI formula for minimization:
    # EI(x) = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    improvement = f_best - mu
    z = improvement / sigma_safe
    
    # Calculate Phi(z) and phi(z)
    # Phi(z) is the CDF of the standard normal distribution
    # phi(z) is the PDF of the standard normal distribution
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    ei = improvement * cdf + sigma_safe * pdf
    
    # EI is always non-negative. 
    # Return EI scores directly; higher EI is better.
    return ei
# EVOLVE-BLOCK-END
