"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Acquisition function based on Upper Confidence Bound (UCB) for minimization.
    
    Parameters:
    mu : array-like, shape (N,)
        Mean of the Gaussian process posterior.
    sigma : array-like, shape (N,)
        Standard deviation of the Gaussian process posterior.
    f_best : float
        Best observed value so far.
        
    Returns:
    scores : array, shape (N,)
        Acquisition scores where higher values indicate more promising candidates.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 0.0)
    
    # UCB for minimization: we want to minimize mu - beta * sigma
    # To get a score where higher is better, we return -mu + beta * sigma
    # A common choice for beta in BO is around 2.0 or scaled by log steps.
    # Given the small number of evaluations (30), a fixed beta is reasonable.
    beta = 2.0
    
    scores = -mu + beta * sigma_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
