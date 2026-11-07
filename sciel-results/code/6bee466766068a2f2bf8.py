"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Look-Ahead Expected Improvement (LAEI) approximation.
    Models the current best value as a random variable with mean mu_best 
    (posterior mean of the best candidate) and variance sigma_best^2 
    (variance of the best candidate).
    """
    # Find the candidate with the lowest posterior mean
    best_idx = np.argmin(mu)
    mu_best = mu[best_idx]
    sigma_best = sigma[best_idx]
    
    # Compute the combined variance
    # D = Y - X, where Y ~ N(mu_best, sigma_best^2) and X ~ N(mu, sigma^2)
    # Var(D) = sigma_best^2 + sigma^2
    combined_var = sigma_best**2 + sigma**2
    combined_std = np.sqrt(combined_var)
    
    # Handle division by zero
    combined_std_safe = np.where(combined_std > 1e-9, combined_std, 1e-9)
    
    # z-score for the difference
    z = (mu_best - mu) / combined_std_safe
    
    # Expected Improvement of the difference
    # E[max(0, D)] = (mu_best - mu) * Phi(z) + combined_std * phi(z)
    ei = (mu_best - mu) * norm.cdf(z) + combined_std * norm.pdf(z)
    
    # Ensure finite scores
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
