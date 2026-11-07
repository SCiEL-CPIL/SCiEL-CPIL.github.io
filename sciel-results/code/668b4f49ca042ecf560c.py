"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement per unit cost.
    
    Parameters:
    mu : array-like, shape (2000,)
        Mean of the GP posterior.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        The lowest observed function value so far.
    cost : array-like, shape (2000,)
        The cost of evaluating each candidate point.
        
    Returns:
    scores : array, shape (2000,)
        Acquisition scores. Higher scores indicate more desirable points.
    """
    # Ensure inputs are numpy arrays for vectorized operations
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Clip sigma to avoid division by zero and numerical instability
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the CDF and phi is the PDF of the standard normal distribution
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # EI should theoretically be non-negative. 
    # Due to floating point errors, it might be slightly negative.
    ei = np.maximum(ei, 0.0)
    
    # Calculate cost-aware score: EI per unit cost
    # We want to maximize EI / cost
    cost_safe = np.maximum(cost, eps)
    scores = ei / cost_safe
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
