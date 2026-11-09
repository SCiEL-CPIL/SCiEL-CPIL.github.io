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
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # Avoid division by zero in sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    # z = (f_best - mu) / sigma
    # If mu < f_best, z is positive, indicating potential improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Where Phi is the standard normal CDF and phi is the standard normal PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure finite values (though EI should be finite if inputs are)
    # Clip very small negative values to 0 due to numerical noise
    ei = np.maximum(ei, 0.0)
    
    return ei
