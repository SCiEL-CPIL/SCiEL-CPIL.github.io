"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization using Lower Confidence Bound (LCB).
    
    For minimization, LCB is defined as mu - kappa * sigma.
    We maximize this score to find the next point to evaluate.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the Gaussian process posterior at candidate points.
    sigma : array-like, shape (2000,)
        Standard deviation of the Gaussian process posterior at candidate points.
    f_best : float
        Best observed value so far (minimum).
    
    Returns:
    --------
    scores : array-like, shape (2000,)
        Acquisition scores, higher values indicate more promising candidates.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # LCB acquisition function for minimization
    # kappa controls the exploration-exploitation trade-off
    # A larger kappa encourages more exploration
    kappa = 2.0
    scores = mu - kappa * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
