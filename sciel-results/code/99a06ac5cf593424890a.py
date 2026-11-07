"""Acquisition function for Bayesian optimization of hartmann 3d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    
    Parameters:
    - mu: array of shape (N,) with posterior mean
    - sigma: array of shape (N,) with posterior standard deviation
    - f_best: scalar, the lowest observed value so far
    
    Returns:
    - scores: array of shape (N,) with acquisition scores (higher is better)
    """
    # Clip sigma to avoid division by zero and ensure positivity
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Improvement = f_best - mu (for minimization, lower mu means higher improvement)
    improvement = f_best - mu
    
    # Standardized improvement
    z = improvement / sigma_safe
    
    # Expected improvement: E[max(0, improvement)]
    # = sigma * (z * Phi(z) + phi(z))
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    # Compute EI carefully to handle numerical issues
    # When z is very negative, EI should be near 0
    # When z is very positive, EI should be close to improvement
    ei = sigma_safe * (z * Phi + phi)
    
    # For very negative z, EI can become numerically unstable, so clip
    # When z < -10, EI is essentially 0
    ei = np.where(z < -10, 0.0, ei)
    
    # Ensure all scores are finite and non-negative
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
