import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Posterior mean at candidate points (in objective units)
    sigma : array-like, shape (2000,)
        Posterior standard deviation at candidate points (in objective units)
    f_best : float
        Best observed function value so far
        
    Returns:
    --------
    scores : array, shape (2000,)
        Acquisition scores; higher values are preferred
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-10)
    
    # Use Expected Improvement (EI) acquisition function
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    # Normalize for numerical stability
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)  # Clip for numerical stability
    
    # Compute EI
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add a small exploration bonus to encourage exploration
    # This helps escape local minima and explore the space better
    # Use a diminishing exploration term based on sigma
    exploration_bonus = 0.1 * sigma
    
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
