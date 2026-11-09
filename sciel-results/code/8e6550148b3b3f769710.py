"""Rewritten acquisition: Expected Improvement (EI)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Compute the Expected Improvement (EI) acquisition function for minimization.
    
    Parameters:
    mu: array of shape (N,) - mean of the GP posterior at candidate points
    sigma: array of shape (N,) - standard deviation of the GP posterior at candidate points
    f_best: float - the best (lowest) observed function value so far
    
    Returns:
    scores: array of shape (N,) - EI values, higher means more promising
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate the improvement: f_best - mu
    # If mu < f_best, the improvement is positive
    improvement = f_best - mu
    
    # Standardize the improvement
    z = improvement / sigma_safe
    
    # EI formula for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the CDF and phi is the PDF of the standard normal distribution
    
    # Calculate Phi(z) and phi(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Compute EI
    ei_scores = improvement * Phi_z + sigma_safe * phi_z
    
    # Handle numerical issues: if sigma is very small, EI should be 0 if mu >= f_best
    # and f_best - mu if mu < f_best. The formula handles this, but let's ensure finiteness.
    scores = np.where(np.isfinite(ei_scores), ei_scores, 0.0)
    
    # EI is always non-negative. If no improvement is expected (mu >= f_best and sigma is small),
    # EI will be close to 0.
    
    return scores
# EVOLVE-BLOCK-END
