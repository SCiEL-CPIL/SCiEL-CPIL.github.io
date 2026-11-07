"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition combining expected improvement and uncertainty
    with adaptive exploration and non-linear cost penalty.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Mean of GP posterior at candidate points.
    sigma : array of shape (2000,)
        Standard deviation of GP posterior at candidate points.
    f_best : float
        Best (lowest) observed function value.
    cost : array of shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Safety: ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values with safe defaults
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-3)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-6)
    
    # Adaptive exploration parameter:
    # When improvement is small relative to uncertainty, explore more.
    # beta increases as (f_best - mu) becomes small compared to sigma.
    improvement = f_best - mu
    uncertainty_ratio = sigma / (np.abs(improvement) + 1e-9)
    
    # Beta starts at 1.0 (pure EI-like) and increases with uncertainty
    # relative to current best improvement
    beta = 1.0 + 0.5 * np.clip(uncertainty_ratio, 0, 5.0)
    
    # Expected Improvement
    z = improvement / sigma
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Uncertainty bonus (UCB-like component)
    ucb = beta * sigma
    
    # Combined score: EI + UCB component
    raw_score = ei + ucb
    
    # Non-linear cost penalty: quadratic-ish to strongly favor cheap regions
    # cost ranges from 1 to 2 (1 + mean of scaled coords in [0,1])
    # Normalize cost to [0, 1] for penalty calculation
    cost_norm = (cost - 1.0) / 1.0  # Now in [0, 1]
    cost_penalty = 1.0 + 0.5 * cost_norm**2
    
    # Final score
    scores = raw_score / cost_penalty
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
