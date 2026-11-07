"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    
    EI is a well-established acquisition function that balances exploration
    and exploitation by maximizing the expected improvement over the current
    best observed value f_best.
    
    For minimization, EI at a point x is:
        EI(x) = E[max(f_best - f(x), 0)]
    
    where f(x) ~ N(mu(x), sigma(x)^2).
    """
    # Handle edge case where sigma is zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the CDF and phi is the PDF of the standard normal
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    scores = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Ensure scores are finite and non-negative
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
