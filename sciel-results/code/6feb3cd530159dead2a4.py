"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Targeted Thompson Sampling with cost-awareness.
    
    Samples an offset that is guaranteed to be better than f_best,
    using a robust estimate of uncertainty (median sigma).
    Scores candidates by (offset - mu) / cost.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean predictions from the GP
    sigma : array-like, shape (2000,)
        Standard deviation predictions from the GP
    f_best : float
        Best observed value so far
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more promising)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-12)
    cost = np.maximum(cost, 1e-12)
    
    # Use median sigma for robustness against outliers in uncertainty
    sigma_scale = np.median(sigma)
    
    # Sample a positive improvement magnitude
    # This ensures the target offset is always better (lower) than f_best
    improvement = np.abs(np.random.normal(0, sigma_scale))
    offset = f_best - improvement
    
    # Score each candidate as (offset - mu) / cost
    # We want to maximize this: low mu (good prediction) and low cost
    # The term (offset - mu) is negative for most points, so we are maximizing
    # how much better the point is than the target offset, penalized by cost.
    # Note: Since offset < f_best, and mu is usually > offset for most points,
    # the numerator is negative. We want the *least negative* value (closest to 0 or positive).
    # Actually, if mu < offset, the score is positive.
    # If mu > offset, the score is negative.
    # We pick the max score.
    scores = (offset - mu) / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
