"""Rewritten acquisition function using a different approach family (Upper Confidence Bound with adaptive exploration)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6D.
    
    Uses the Expected Improvement (EI) acquisition function.
    EI directly maximizes the expected reduction in the objective value.
    This approach is often more robust than UCB as it provides a non-linear
    balance between exploitation (low mean) and exploration (high uncertainty).
    
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
        Acquisition scores; higher means more promising to evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero or invalid inputs to norm
    # A small epsilon prevents numerical instability
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate the improvement over the current best
    # For minimization, improvement is f_best - mu
    improvement = f_best - mu
    
    # Calculate the Z-score for the improvement
    # z = (f_best - mu) / sigma
    z = improvement / sigma
    
    # Expected Improvement formula:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Where Phi is the CDF of the standard normal and phi is the PDF
    
    # Use scipy.stats.norm for accurate CDF and PDF
    # norm.cdf(z) gives Phi(z)
    # norm.pdf(z) gives phi(z)
    
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    scores = improvement * cdf + sigma * pdf
    
    # Ensure all scores are finite
    # EI should always be non-negative for valid inputs, but numerical errors can occur
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
