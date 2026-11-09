# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior
    f_best : float
        Best observed function value
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Add small epsilon to avoid division by zero
    sigma_safe = sigma + 1e-10
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Clip negative EI to zero (EI should be non-negative)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost to get EI per unit cost
    scores_ei = ei / (cost + 1e-10)
    
    # Uncertainty bonus: prioritize points with high sigma but reasonable cost
    # This helps escape local minima in rastrigin
    scores_unc = sigma / (cost + 1e-10)
    
    # Combine: use a weighted combination to balance exploitation and exploration
    # Higher weight on EI when we're close to optimum, more exploration early
    # Use a simple heuristic: 0.8 * EI + 0.2 * uncertainty
    scores = 0.8 * scores_ei + 0.2 * scores_unc
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
