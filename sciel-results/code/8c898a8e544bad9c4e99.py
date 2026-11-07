"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track start time for deadline guard
_START_TIME = time.time()
_DEADLINE_MARGIN = 3.0  # seconds before the 330s limit


def _elapsed():
    return time.time() - _START_TIME


def _remaining_time():
    return 330.0 - _elapsed() - _DEADLINE_MARGIN


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Args:
        mu: (2000,) array of GP mean predictions in objective units
        sigma: (2000,) array of GP std deviations in objective units
        f_best: lowest observed function value
        cost: (2000,) array of evaluation costs for each candidate
    
    Returns:
        (2000,) array of scores (higher = evaluate next)
    """
    # Deadline guard: if we're running low on time, return a safe fallback
    if _remaining_time() < 0.1:
        # Return a simple fallback: favor low cost, low mu
        return -(mu + 0.1 * cost)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Compute expected improvement (EI)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Adaptive kappa based on how far we are from the optimum
    # As f_best approaches 0 (or the known minimum), reduce exploration
    # Use the cost to infer progress: higher total cost means more evaluations
    mean_cost = np.mean(cost)
    # Estimate progress: if f_best is close to 0, we're near the optimum
    # Scale kappa: start with moderate exploration, reduce as we improve
    # f_best is the best value seen; for these functions min is 0 or -3.86
    # Use a soft measure of progress
    progress = np.clip(-f_best / (1.0 + np.abs(f_best)), 0.0, 1.0)
    
    # Adaptive weights: blend EI with a cost-adjusted exploration term
    # Higher cost regions should be penalized more
    # Use a power law to control the trade-off
    cost_penalty = np.power(cost, 1.2)
    
    # Exploration term: favor uncertain regions
    # Scale by (1 - progress) to reduce exploration as we converge
    exploration_weight = 0.3 * (1.0 - 0.5 * progress)
    exploitation_weight = 1.0
    
    # Combine: weighted EI per unit cost + exploration bonus
    # The EI is in objective units, so normalize by a scale
    # Use the std of mu to normalize EI
    mu_scale = np.std(mu) + 1e-8
    ei_normalized = ei / mu_scale
    
    # Cost-adjusted score
    # We want high EI, low cost, and some exploration
    # score = (ei_normalized * exploitation_weight + exploration_weight * sigma / mu_scale) / cost_penalty
    score = (ei_normalized * exploitation_weight + 
             exploration_weight * sigma / mu_scale) / cost_penalty
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
