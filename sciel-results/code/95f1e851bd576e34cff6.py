import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Computes Expected Improvement (EI) scores for minimization.
    
    Parameters:
    mu (np.ndarray): Mean of the Gaussian Process posterior. Shape (N,).
    sigma (np.ndarray): Standard deviation of the GP posterior. Shape (N,).
    f_best (float): The best (lowest) observed function value so far.
    
    Returns:
    np.ndarray: EI scores. Higher scores indicate more promising points.
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma = np.clip(sigma, 1e-10, None)
    
    # Calculate the normalized improvement z
    z = (f_best - mu) / sigma
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Using the fact that f_best - mu = z * sigma:
    # EI = sigma * (z * Phi(z) + phi(z))
    
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    scores = sigma * (z * cdf_z + pdf_z)
    
    # Handle any non-finite values (shouldn't happen with clipped sigma, but just in case)
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
