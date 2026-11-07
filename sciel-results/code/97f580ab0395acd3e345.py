"""Cost-aware acquisition function for Bayesian optimization (three hump camel 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function combining cost-aware Expected Improvement with a
    decaying UCB exploration term to balance exploitation and exploration.
    
    Parameters
    ----------
    mu : array_like, shape (2000,)
        GP mean predictions.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed function value so far.
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate (1 + mean of scaled coordinates).
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware component: use a softened cost penalty instead of hard division
    # This reduces bias against expensive regions where the minimum may reside
    # cost_penalty = 1.0 + 0.2 * (cost - 1.0)  # Gentle linear penalty
    # But let's use a more principled approach: EI / (cost^alpha) with small alpha
    
    # Use a power law cost adjustment: EI / cost^0.5
    # This is less aggressive than EI/cost but still cost-aware
    cost_factor = np.power(cost, 0.5)
    cost_aware_ei = ei / cost_factor
    
    # UCB exploration term with decay
    # The beta parameter controls exploration; we can estimate elapsed evaluations
    # from the cost array (higher cost means more evaluations have occurred)
    # Use mean cost as a proxy for progress
    mean_cost = np.mean(cost)
    
    # Beta decays as we collect more data
    # Early on (low mean cost), explore more; later, exploit more
    # Start with beta ~ 2.0 and decay to ~ 0.5
    beta = 2.0 * np.exp(-0.3 * (mean_cost - 1.0))
    
    ucb_term = beta * sigma_safe
    
    # Combine: cost-aware EI + UCB exploration
    scores = cost_aware_ei + ucb_term
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
