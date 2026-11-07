"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Computes cost-aware Expected Improvement with robust numerical handling.
    
    Args:
        mu: array of shape (N,) predicted mean
        sigma: array of shape (N,) predicted std deviation
        f_best: float, best value found so far
        cost: array of shape (N,) evaluation cost for each candidate
        
    Returns:
        scores: array of shape (N,) acquisition values (higher is better)
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to a small positive value to prevent division by zero
    sigma_clipped = np.maximum(sigma, 1e-9)
    
    # Clip cost to a small positive value
    cost_clipped = np.maximum(cost, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_clipped
    
    # Calculate PDF and CDF
    pdf = norm.pdf(z)
    cdf = norm.cdf(z)
    
    # Expected Improvement
    ei = (f_best - mu) * cdf + sigma_clipped * pdf
    
    # Cost-aware score
    scores = ei / cost_clipped
    
    # Replace non-finite values with a very small negative number
    # This ensures the returned array contains only finite values
    scores = np.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)
    
    return scores
# EVOLVE-BLOCK-END
