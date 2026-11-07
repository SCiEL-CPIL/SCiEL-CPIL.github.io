"""Rewritten program using Probability of Improvement (POI) acquisition function."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Probability of Improvement (POI) acquisition function.
    
    Parameters:
    -----------
    mu : array-like, shape (n,)
        Mean of the GP posterior at candidate points.
    sigma : array-like, shape (n,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far (minimum, since we're minimizing).
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        POI scores for each candidate. Higher values indicate more promising candidates.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute the z-score: (f_best - mu) / sigma
    # Since we're minimizing, we want mu < f_best
    z = (f_best - mu) / sigma_safe
    
    # POI = P(f(x) < f_best) = Phi(z)
    # where Phi is the standard normal CDF
    poi_scores = norm.cdf(z)
    
    # Ensure all scores are finite
    poi_scores = np.nan_to_num(poi_scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    return poi_scores
# EVOLVE-BLOCK-END
