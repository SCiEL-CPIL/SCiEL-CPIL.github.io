"""Rewritten program using a different acquisition approach: Cost-Aware Expected Improvement (EI)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function.
    
    Parameters:
    - mu: array of shape (2000,) - GP mean predictions
    - sigma: array of shape (2000,) - GP standard deviation predictions
    - f_best: scalar - best (lowest) observed function value so far
    - cost: array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
    - scores: array of shape (2000,) - acquisition scores (higher = better)
    """
    # Ensure sigma is non-negative and handle zeros to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate z-scores: (f_best - mu) / sigma
    # This represents how many standard deviations the mean is below the current best
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Phi is the standard normal CDF, phi is the standard normal PDF
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    EI = (f_best - mu) * Phi + sigma_safe * phi
    
    # EI should theoretically be non-negative, but clamp to 0 to handle numerical noise
    EI = np.maximum(EI, 0.0)
    
    # Cost-aware score: Expected Improvement per unit cost
    scores = EI / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
