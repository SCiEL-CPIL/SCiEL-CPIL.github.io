import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Compute Expected Improvement (EI) scores for candidate points.
    
    Parameters:
    mu : np.ndarray
        Mean of the GP posterior at candidate points, shape (N,)
    sigma : np.ndarray
        Standard deviation of the GP posterior at candidate points, shape (N,)
    f_best : float
        Current best (lowest) observed value.
        
    Returns:
    np.ndarray
        EI scores for each candidate, shape (N,)
    """
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Where Phi is CDF and phi is PDF of standard normal
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure finite values
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    return ei
