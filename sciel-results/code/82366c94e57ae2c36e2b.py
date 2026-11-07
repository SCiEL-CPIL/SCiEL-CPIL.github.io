import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    Uses Expected Improvement (EI) with xi=0 for pure exploitation.
    EI measures the expected improvement over the current best f_best.
    Higher EI score means more likely to find a better solution.
    """
    # For minimization, improvement is when f(x) < f_best
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi and phi are the CDF and PDF of the standard normal
    
    # Handle cases where sigma is very small
    safe_sigma = np.maximum(sigma, 1e-10)
    
    # Standardized improvement: z = (f_best - mu) / sigma
    z = (f_best - mu) / safe_sigma
    
    # Compute EI using the standard normal CDF and PDF
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + safe_sigma * phi_z
    
    # EI is non-negative; higher means more expected improvement
    # Ensure all values are finite
    scores = np.where(np.isfinite(ei), ei, 0.0)
    
    return scores
