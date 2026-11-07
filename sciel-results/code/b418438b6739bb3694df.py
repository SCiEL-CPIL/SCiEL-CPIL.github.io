"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function.
    
    This function computes a score for each candidate point based on the
    Expected Improvement for minimization, normalized by the evaluation cost.
    Higher scores indicate candidates that are expected to improve the
    best observed value per unit cost.
    
    Args:
        mu: Array of GP mean predictions, shape (N,).
        sigma: Array of GP standard deviations, shape (N,).
        f_best: Scalar, the lowest observed value so far.
        cost: Array of evaluation costs for each candidate, shape (N,).
        
    Returns:
        Array of finite scores, shape (N,).
    """
    # Ensure inputs are numpy arrays for vectorized operations
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Avoid division by zero in Z calculation
    # If sigma is 0, the posterior is a delta function.
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Standardized improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the CDF and phi is the PDF of the standard normal distribution
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Normalize by cost to account for the budget constraint
    # Use a small epsilon to prevent division by zero
    safe_cost = np.maximum(cost, 1e-6)
    
    # The acquisition score is the expected improvement per unit cost
    scores = ei / safe_cost
    
    # Ensure all scores are finite to satisfy the task contract
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
