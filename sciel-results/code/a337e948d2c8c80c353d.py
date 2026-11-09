"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Probability of Improvement (PoI)
    weighted by inverse cost with a small GP uncertainty bonus.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions at 2000 candidate points.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions at 2000 candidate points.
    f_best : float
        Best (lowest) observed objective value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Probability of Improvement: P(f(x) < f_best)
    # Using the standard normal CDF
    z = (f_best - mu) / sigma
    poi = norm.cdf(z)
    
    # Weight the PoI by the expected magnitude of improvement (f_best - mu)
    # This creates a "Probability-Weighted Improvement" score, which is more
    # informative than raw PoI as it distinguishes between small and large improvements.
    weighted_poi = poi * (f_best - mu)
    
    # Uncertainty bonus to encourage exploration
    # Increased weight (1.0) compared to previous (0.1) to balance exploitation
    # and exploration more effectively, similar to UCB.
    uncertainty_bonus = 1.0 * sigma
    
    # Cost-adjusted score: prioritize high value per unit cost
    scores = (weighted_poi + uncertainty_bonus) / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
