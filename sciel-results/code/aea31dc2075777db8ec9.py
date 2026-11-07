# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-penalized Expected Improvement.
    
    Parameters
    ----------
    mu : array-like, shape (n,)
        GP mean predictions.
    sigma : array-like, shape (n,)
        GP standard deviation predictions.
    f_best : float
        Best observed value so far.
    cost : array-like, shape (n,)
        Cost of evaluating each candidate point.
    
    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition values; higher is better.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero and numerical instability
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Standard normal z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost normalization: cost is in [1, 2]
    # We penalize deviation from the minimum possible cost (1.0)
    # This ensures the cheapest points have 0 penalty.
    # lambda controls the strength of the cost aversion.
    # A value of 0.3 suggests that a cost increase of 1.0 (from 1.0 to 2.0)
    # reduces the acquisition score by 0.3 units of EI.
    lambda_cost = 0.3
    cost_penalty = lambda_cost * (cost - 1.0)
    
    scores = ei - cost_penalty
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
