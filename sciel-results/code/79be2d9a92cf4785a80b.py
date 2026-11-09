import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    Uses Expected Improvement (EI) for minimization.
    
    EI(x) = E[max(f_best - f(x), 0)]
    For f(x) ~ N(mu, sigma^2), EI can be computed analytically:
    EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    This strategy prioritizes points where the expected reduction in the best 
    observed value is highest, balancing exploration and exploitation naturally
    based on the posterior uncertainty.
    """
    # Ensure arrays are floats
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Calculate the improvement over the current best
    # Note: We are minimizing, so improvement is f_best - f(x)
    improvement = f_best - mu
    
    # Standard normal CDF and PDF
    # z = (f_best - mu) / sigma
    z = improvement / sigma_safe
    
    # Handle edge cases where z might be extremely large/small
    # norm.cdf and norm.pdf are stable, but let's ensure no NaNs
    cdf_vals = norm.cdf(z)
    pdf_vals = norm.pdf(z)
    
    # EI formula for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei_scores = improvement * cdf_vals + sigma_safe * pdf_vals
    
    # Ensure scores are finite and non-negative (EI is always >= 0)
    ei_scores = np.where(np.isfinite(ei_scores), ei_scores, 0.0)
    ei_scores = np.maximum(ei_scores, 0.0)
    
    return ei_scores
