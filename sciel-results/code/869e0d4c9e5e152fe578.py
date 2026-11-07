import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_DEADLINE = 300.0  # Safety margin before 330s limit

def acquisition(mu, sigma, f_best):
    """
    Computes the Expected Improvement (EI) acquisition function.
    
    Parameters:
    mu : np.ndarray
        Mean of the Gaussian Process posterior, shape (N,)
    sigma : np.ndarray
        Standard deviation of the Gaussian Process posterior, shape (N,)
    f_best : float
        The best (lowest) observed function value so far.
        
    Returns:
    np.ndarray
        Acquisition scores for each candidate, shape (N,). Higher is better.
    """
    # Deadline guard
    if time.time() - _start_time > _DEADLINE:
        # Fallback to a simple UCB if we are running out of time
        sigma_safe = np.maximum(sigma, 1e-8)
        return -mu + 2.0 * sigma_safe

    # Ensure sigma is positive to avoid division by zero or log(0)
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    # Using scipy for numerical stability
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure finiteness
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
