"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    
    For minimization, the improvement at a point x is:
    I(x) = max(0, f_best - f(x))
    
    Under the GP posterior, f(x) ~ N(mu(x), sigma(x)^2)
    
    The expected improvement is:
    E[I(x)] = (f_best - mu(x)) * Phi((f_best - mu(x)) / sigma(x)) 
             + sigma(x) * phi((f_best - mu(x)) / sigma(x))
    
    where Phi is the standard normal CDF and phi is the standard normal PDF.
    
    This is a well-established acquisition function that balances exploration
    and exploitation more effectively than UCB in many cases.
    """
    from scipy.stats import norm
    
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-10)  # Avoid division by zero
    
    # Compute the standardized distance
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    # E[I] = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF and phi is PDF of standard normal
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    scores = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
