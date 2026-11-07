"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    
    For minimization, EI measures the expected improvement over f_best:
    EI(x) = E[max(f_best - f(x), 0)]
    
    Given f(x) ~ N(mu(x), sigma(x)^2):
    EI(x) = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    We return scores where higher means evaluate next, so we return EI directly.
    """
    # Ensure sigma is positive and avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute the standardized distance
    d = (f_best - mu) / sigma_safe
    
    # Standard normal CDF and PDF
    cdf = norm.cdf(d)
    pdf = norm.pdf(d)
    
    # Expected Improvement for minimization
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Ensure scores are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
