"""Rewritten acquisition function using Expected Improvement (EI) for minimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6D.
    
    Uses Expected Improvement (EI) for minimization. This approach balances
    exploration and exploitation by estimating the expected amount by which
    a candidate point improves upon the current best value.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior at candidate points (in objective units, lower is better).
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Lowest observed objective value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (Expected Improvement); higher means more promising to evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate the improvement term: how much better the best is than the mean
    # For minimization, improvement is f_best - mu if mu < f_best
    improvement = f_best - mu
    
    # Calculate the z-score
    z = improvement / sigma
    
    # Expected Improvement formula for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    scores = improvement * Phi_z + sigma * phi_z
    
    # Ensure all scores are finite and non-negative (EI should be >= 0)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
