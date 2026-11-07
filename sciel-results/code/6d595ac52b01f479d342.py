"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function.
    
    Unlike Expected Improvement (EI), which focuses on the expected magnitude of 
    improvement, UCB is optimistic about the potential of high-uncertainty regions. 
    This is structurally different as it adds an exploration bonus proportional 
    to the standard deviation, rather than integrating the improvement distribution.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the Gaussian Process posterior.
    sigma : array-like, shape (2000,)
        Standard deviation of the Gaussian Process posterior.
    f_best : float
        The best (lowest) observed value so far.
        
    Returns:
    --------
    scores : array, shape (2000,)
        UCB scores. Higher is better (more promising).
    """
    # Use a relatively high beta to encourage exploration in early/mid stages.
    # For Ackley function which is multimodal, strong exploration is needed
    # to escape local minima and find the global minimum at the origin.
    # EI effectively uses a beta that varies but is often smaller.
    # We fix beta to a value that balances exploitation (-mu) and exploration (beta * sigma).
    
    beta = 2.5
    
    # Handle numerical stability for sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB score: we want to minimize f, so we maximize -mu + beta * sigma
    scores = -mu + beta * sigma_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
