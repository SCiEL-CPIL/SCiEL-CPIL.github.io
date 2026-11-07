"""Cost-aware acquisition function for Bayesian optimization (styblinski tang 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using a Pareto-frontier based approach.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : ndarray, shape (2000,)
        Evaluation costs for each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    start_time = time.time()
    deadline = 320.0  # Safety margin before 330s kill
    
    n = len(mu)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute expected improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (it should be, but guard against numerical issues)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: EI per unit cost
    cost_safe = np.maximum(cost, 1e-9)
    ei_per_cost = ei / cost_safe
    
    # Pareto-frontier based scoring:
    # We want to find points that are on the cost-efficient frontier
    # where marginal reduction in predicted value per unit cost is maximized.
    
    # Sort by predicted value (lower is better)
    pred_value = mu  # Lower is better for the objective
    
    # For Pareto front: a point dominates another if it has both
    # lower predicted value AND lower cost
    # We want points on the non-dominated frontier
    
    # Efficient approach: sort by cost, then track minimum predicted value
    # Points where predicted value is at a new minimum when sorted by cost
    # are on the Pareto front
    
    # Sort indices by cost (ascending)
    cost_order = np.argsort(cost)
    sorted_cost = cost[cost_order]
    sorted_mu = mu[cost_order]
    
    # Find Pareto front: points where mu is strictly less than all previous
    # points with lower or equal cost
    is_pareto = np.zeros(n, dtype=bool)
    min_mu_so_far = np.inf
    for i in range(n):
        if sorted_mu[i] < min_mu_so_far:
            is_pareto[cost_order[i]] = True
            min_mu_so_far = sorted_mu[i]
    
    # Score: combine EI per cost with Pareto front bonus
    # Points on the Pareto front get a boost
    
    # Check time
    if time.time() > deadline - 5.0:
        # Return EI per cost as fallback
        scores = ei_per_cost.copy()
        scores[~np.isfinite(scores)] = 0.0
        return scores
    
    # Create score: base is EI per cost, with Pareto front points boosted
    # Use a smooth weighting: points closer to the Pareto front (in terms of
    # being on the efficient frontier) get higher scores
    
    # Alternative: rank-based scoring
    # Rank by EI per cost (higher is better)
    ei_per_cost_rank = np.argsort(np.argsort(ei_per_cost))
    
    # Rank by predicted value (lower is better, so reverse)
    mu_rank = np.argsort(np.argsort(-mu))
    
    # Rank by cost (lower is better)
    cost_rank = np.argsort(np.argsort(-cost))
    
    # Combined score: weighted combination of ranks
    # Higher EI per cost, lower mu, lower cost -> better
    score = (ei_per_cost_rank + mu_rank + cost_rank) / 3.0
    
    # Add Pareto front bonus
    pareto_bonus = np.where(is_pareto, 1.0, 0.0)
    
    # Final score: emphasize Pareto front points
    final_score = score + 0.5 * pareto_bonus
    
    # Ensure finite values
    final_score[~np.isfinite(final_score)] = 0.0
    
    return final_score
# EVOLVE-BLOCK-END
