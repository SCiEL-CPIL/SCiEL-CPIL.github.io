import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    mu: array of posterior means
    sigma: array of posterior standard deviations
    f_best: current best observed value (minimum)
    Returns: array of EI scores (higher is better)
    """
    # Avoid division by zero in sigma
    sigma_safe = np.where(sigma > 1e-10, sigma, 1e-10)
    
    # Calculate Z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate EI
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: For minimization, we expect improvement if mu < f_best.
    # The standard EI formula for minimization is:
    # EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x))/sigma(x)) + sigma(x) * phi((f_best - mu(x))/sigma(x))
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure finite scores
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
