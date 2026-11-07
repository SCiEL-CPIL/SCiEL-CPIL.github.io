import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    
    Parameters:
    -----------
    mu : array-like, shape (n,)
        Mean of the Gaussian process posterior at candidate points.
    sigma : array-like, shape (n,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        EI scores where higher values indicate more promising candidates.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle edge cases
    n = mu.shape[0]
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # EI for minimization: E[min(f_best - f, 0)]
    # where f ~ N(mu, sigma^2)
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    #     when sigma > 0
    # EI = max(f_best - mu, 0) when sigma == 0
    
    # Compute z = (f_best - mu) / sigma
    # We need to be careful with sigma near zero
    
    # Create a safe version of sigma for division
    sigma_safe = np.where(sigma > 1e-10, sigma, 1.0)
    
    z = (f_best - mu) / sigma_safe
    
    # EI formula
    # For points where sigma is effectively zero, EI is just max(f_best - mu, 0)
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # Where sigma was very small, use the deterministic improvement
    small_sigma = sigma <= 1e-10
    ei[small_sigma] = np.maximum(f_best - mu[small_sigma], 0.0)
    
    # Ensure all scores are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
