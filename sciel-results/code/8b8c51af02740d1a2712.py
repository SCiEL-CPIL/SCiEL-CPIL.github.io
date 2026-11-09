"""Cost-aware acquisition function for Bayesian optimization (three hump camel 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function that combines expected improvement
    with an exploration bonus to escape local optima.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions
    sigma : array-like, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed function value so far
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Ensure arrays are float64 for numerical stability
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-normalized EI
    cost_safe = np.maximum(cost, 1e-9)
    ei_cost = ei / cost_safe
    
    # Exploration bonus: reward high uncertainty
    # Scale the exploration term to be comparable to EI
    # Use a weighted combination that balances exploitation and exploration
    # The exploration bonus helps escape local optima in the three-hump camel landscape
    
    # Normalize sigma for the exploration term
    sigma_max = np.max(sigma_safe)
    if sigma_max > 0:
        sigma_norm = sigma_safe / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # Exploration bonus with a tunable weight
    # A moderate weight helps exploration without dominating the EI signal
    exploration_weight = 0.3
    exploration_bonus = exploration_weight * sigma_norm / cost_safe
    
    # Combined score
    scores = ei_cost + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
