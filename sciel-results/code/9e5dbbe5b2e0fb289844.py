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
    
    # UCB parameter: balance between exploitation and exploration
    # A common heuristic is beta=2 or beta=1. For Branin, which is multimodal,
    # a higher beta might help escape local minima, but too high leads to random search.
    # We start with beta=1.5 as a reasonable middle ground.
    beta = 1.5
    
    # For minimization, we want to minimize mu - beta * sigma.
    # To fit the interface where "higher score means evaluate next",
    # we can return the negative of the UCB value, or simply use a form that rewards low mu and high sigma.
    # Standard UCB for minimization: q(x) = mu(x) - beta * sigma(x)
    # We want to maximize the score. So score = - (mu - beta * sigma) = -mu + beta * sigma.
    
    scores = -mu + beta * sigma_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
