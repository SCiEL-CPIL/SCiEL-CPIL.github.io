import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of hartmann 6d.
    Uses Expected Improvement (EI) to balance exploration and exploitation.
    
    Parameters:
    mu: array of shape (2000,) - posterior mean
    sigma: array of shape (2000,) - posterior standard deviation
    f_best: float - best observed value (minimum)
    
    Returns:
    scores: array of shape (2000,) - acquisition scores (higher is better)
    """
    start_time = time.time()
    deadline = 325.0

    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-9)

    # Compute Expected Improvement
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # Higher EI means higher expected improvement over f_best
    
    z = (f_best - mu) / sigma_safe
    
    # Calculate CDF and PDF
    cdf_vals = norm.cdf(z)
    pdf_vals = norm.pdf(z)
    
    # Calculate EI
    ei_scores = (f_best - mu) * cdf_vals + sigma_safe * pdf_vals
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(ei_scores), ei_scores, -np.inf)
    
    # Check deadline
    if time.time() - start_time > deadline:
        return scores

    return scores
