import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Combines Expected Improvement (EI) with a UCB-style exploration term,
    discounted by a soft cost factor.
    """
    t_start = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    sigma = np.maximum(sigma, 1e-8)
    
    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    improvement = np.maximum(f_best - mu, 0.0)
    z = improvement / sigma
    z = np.clip(z, -10, 10)
    
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    
    # UCB exploration term
    # This adds a bonus proportional to uncertainty, encouraging exploration
    # in high-sigma regions even if the mean is not currently promising.
    beta = 0.5
    ucb_exploration = beta * sigma
    
    # Combine EI and exploration
    # EI is the primary driver for exploitation.
    # UCB exploration ensures we don't get stuck in local minima by
    # always having a non-zero incentive to probe high-uncertainty areas.
    base_score = ei + ucb_exploration
    
    # Apply soft cost penalty
    # Cost is in [1, 2]. We want to prefer cheaper evaluations.
    # Using cost^-alpha provides a smooth, multiplicative discount.
    # alpha = 0.5 is a moderate penalty (sqrt(2) ~ 1.41 max discount).
    alpha = 0.5
    cost_factor = np.power(cost, -alpha)
    
    scores = base_score * cost_factor
    
    # Deadline guard: if computation is unexpectedly slow, return a safe fallback
    if time.time() - t_start > 2.0:
        # Fallback: simple UCB without cost
        return ucb_exploration + ei
        
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
