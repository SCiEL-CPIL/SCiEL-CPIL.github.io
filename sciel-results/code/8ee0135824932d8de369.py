"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Probability of Improvement (PI) acquisition function.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        Mean of the Gaussian process posterior at the candidate points.
    sigma : array_like, shape (2000,)
        Standard deviation of the Gaussian process posterior at the candidate points.
    f_best : float
        The lowest observed value so far.
        
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores for each candidate point. Higher scores indicate
        points that are more likely to be evaluated next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Probability of Improvement: P(f(x) < f_best)
    # For a normal distribution N(mu, sigma^2), 
    # P(X < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma
    pi_scores = norm.cdf(z)
    
    # Return the PI scores. Higher values mean higher probability of improvement.
    return pi_scores
# EVOLVE-BLOCK-END
