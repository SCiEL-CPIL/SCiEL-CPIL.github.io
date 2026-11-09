"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a quantile-based heuristic.
    Ranks candidates by the ratio of their posterior probability of
    exceeding the current best to their cumulative cost, with a bias
    toward lower-cost regions and higher uncertainty (sigma).
    
    Args:
        mu: (2000,) array of GP mean predictions.
        sigma: (2000,) array of GP standard deviations.
        f_best: scalar, the best observed value so far.
        cost: (2000,) array of costs for each candidate.
        
    Returns:
        (2000,) array of scores. Higher is better.
    """
    # Ensure arrays are float
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Prevent division by zero in sigma
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate Z-score for P(f(x) < f_best)
    # Since we are minimizing, improvement is when f(x) is lower than f_best.
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Prevent division by zero in cost
    cost_safe = np.maximum(cost, 1e-8)
    
    # Quantile-based scoring: combine PI with uncertainty and cost
    # Use a weighted combination that emphasizes:
    # 1. High probability of improvement
    # 2. High uncertainty (sigma) - to explore
    # 3. Low cost - to be efficient
    
    # Normalize sigma to [0, 1] range for scaling
    sigma_min = np.min(sigma_safe)
    sigma_max = np.max(sigma_safe)
    if sigma_max > sigma_min:
        sigma_norm = (sigma_safe - sigma_min) / (sigma_max - sigma_min)
    else:
        sigma_norm = np.ones_like(sigma_safe)
    
    # Normalize cost to [0, 1] range (lower cost = higher score)
    cost_min = np.min(cost_safe)
    cost_max = np.max(cost_safe)
    if cost_max > cost_min:
        cost_norm = (cost_safe - cost_min) / (cost_max - cost_min)
    else:
        cost_norm = np.zeros_like(cost_safe)
    
    # Score: weighted combination of PI, sigma, and inverse cost
    # Higher PI, higher sigma, and lower cost should yield higher scores
    scores = pi * (1.0 + sigma_norm) / (cost_safe ** 0.5)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
