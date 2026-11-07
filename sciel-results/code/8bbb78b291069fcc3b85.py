"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-normalized Lower Confidence Bound (LCB) acquisition function.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean predictions from GP in objective units.
    sigma : array-like, shape (2000,)
        Standard deviation predictions from GP in objective units.
    f_best : float
        Best (lowest) observed function value.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition values (higher = evaluate next).
    """
    # Ensure arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Compute expected improvement (EI)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Compute LCB with a dynamic exploration coefficient
    # The exploration term is scaled by remaining budget consideration
    # In 8D, variance can be misleading, so we use a moderate exploration factor
    # and normalize by cost to account for expensive evaluations
    
    # Dynamic beta: balance exploitation and exploration
    # Use a fixed beta that works well for 8D problems
    beta = 1.5
    
    # LCB: lower confidence bound (we want to minimize, so lower is better)
    # Acquisition should be high when we expect improvement
    # LCB = mu - beta * sigma (this is a lower bound on the function value)
    # We want to select points with low LCB, so acquisition = -LCB / cost
    lcb = mu - beta * sigma
    lcb_score = -lcb / cost
    
    # Combine EI and LCB: use a weighted combination
    # EI captures expected improvement, LCB captures uncertainty
    # Normalize both to have similar scales
    ei_score = ei / cost
    
    # Weighted combination: favor EI slightly more for exploitation
    # but include LCB for exploration
    alpha = 0.6  # weight for EI
    scores = alpha * ei_score + (1 - alpha) * lcb_score
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
