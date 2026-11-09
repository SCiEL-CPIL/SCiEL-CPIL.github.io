"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) for minimization.
    
    EI(x) = E[max(0, f_best - f(x))]
    where f(x) ~ N(mu(x), sigma(x)^2)
    
    This is a structural change from UCB to EI, which is a non-linear
    combination of the mean and standard deviation, weighted by the
    probability of improvement.
    """
    # Handle zero or near-zero sigma to avoid division by zero
    sigma_safe = np.where(sigma > 1e-10, sigma, 1e-10)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the CDF and phi is the PDF of the standard normal distribution
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Safety check: ensure finite scores
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=0.0)
    
    # Ensure non-negative EI
    ei = np.maximum(ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
