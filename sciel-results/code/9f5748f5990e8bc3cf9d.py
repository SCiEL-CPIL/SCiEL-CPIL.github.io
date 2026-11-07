# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: 330s wall-clock limit, use 325s as safety margin
_DEADLINE = 325.0
_START_TIME = time.time()

def _check_deadline():
    """Return True if we should abort and return current best."""
    return (time.time() - _START_TIME) > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Pareto-frontier based acquisition for cost-aware Bayesian optimization.
    
    Scores candidates by their rank in a non-dominated sort of (predicted value, evaluation cost).
    Points on the cost-efficient frontier where marginal reduction in predicted value per unit cost
    is maximized get the highest scores.
    """
    # Check deadline first
    if _check_deadline():
        return np.zeros_like(mu, dtype=np.float64)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Clip to reasonable ranges
    sigma = np.clip(sigma, 1e-8, 1e4)
    cost = np.clip(cost, 1.0, 2.0)
    
    n = len(mu)
    
    # Compute the predicted lower bound (optimistic prediction) for each candidate
    # Use a quantile of the predictive distribution: mu - k*sigma
    # This represents the best-case outcome if we evaluate at this point
    k = 1.5  # exploration coefficient
    pred_value = mu - k * sigma
    
    # For Pareto front: we want to minimize both pred_value and cost
    # A point is non-dominated if no other point has both lower pred_value and lower cost
    
    # Efficient non-dominated sort using vectorized operations
    # First, sort by pred_value, then find the frontier
    sort_idx = np.argsort(pred_value)
    sorted_pred = pred_value[sort_idx]
    sorted_cost = cost[sort_idx]
    
    # Find non-dominated points: scan sorted by pred_value,
    # a point is non-dominated if its cost is lower than all previously seen points
    # (since pred_value is already sorted ascending)
    frontier_mask = np.zeros(n, dtype=bool)
    
    # Track the minimum cost seen so far in the sorted order
    min_cost_so_far = np.inf
    for i in range(n):
        if sorted_cost[i] < min_cost_so_far:
            frontier_mask[sort_idx[i]] = True
            min_cost_so_far = sorted_cost[i]
    
    # For points on the Pareto frontier, score by marginal improvement per unit cost
    # For off-frontier points, score lower based on their distance from the frontier
    
    # Compute the "efficiency" score: how much improvement per unit cost
    # On the frontier, higher efficiency = better
    # We use: (f_best - pred_value) / cost as the marginal value
    # But we need to handle the frontier structure properly
    
    # Assign scores based on frontier rank and efficiency
    scores = np.zeros(n, dtype=np.float64)
    
    # For frontier points: score proportional to their efficiency
    frontier_indices = np.where(frontier_mask)[0]
    
    if len(frontier_indices) > 0:
        frontier_eff = (f_best - pred_value[frontier_indices]) / cost[frontier_indices]
        frontier_eff = np.maximum(frontier_eff, 0.0)
        
        # Normalize frontier efficiency to [0, 1] range
        eff_min = np.min(frontier_eff)
        eff_max = np.max(frontier_eff)
        if eff_max > eff_min:
            frontier_norm = (frontier_eff - eff_min) / (eff_max - eff_min)
        else:
            frontier_norm = np.ones_like(frontier_eff)
        
        # Frontier points get high scores in [1, 2]
        scores[frontier_indices] = 1.0 + frontier_norm
    
    # For non-frontier points: score based on how close they are to the frontier
    # Use a penalty based on the cost of the dominating frontier point
    non_frontier_indices = np.where(~frontier_mask)[0]
    
    if len(non_frontier_indices) > 0:
        # For each non-frontier point, find the best frontier point that dominates it
        # and assign a score based on the gap
        # Simplified: score = 1.0 - normalized_distance_to_frontier
        
        # Compute a "dominance penalty" for each non-frontier point
        # The penalty is how much worse it is compared to the best frontier point
        # in terms of the scalarized objective
        
        # Use the minimum frontier efficiency as baseline
        if len(frontier_indices) > 0:
            best_frontier_eff = np.max(frontier_eff)
        else:
            best_frontier_eff = 0.0
        
        non_frontier_eff = (f_best - pred_value[non_frontier_indices]) / cost[non_frontier_indices]
        non_frontier_eff = np.maximum(non_frontier_eff, 0.0)
        
        # Score non-frontier points in [0, 1) range
        if best_frontier_eff > 0:
            non_frontier_scores = np.minimum(non_frontier_eff / best_frontier_eff, 1.0) * 0.9
        else:
            non_frontier_scores = np.zeros_like(non_frontier_eff)
        
        scores[non_frontier_indices] = non_frontier_scores
    
    # Add a small exploration boost for high-sigma points to encourage exploration
    # in unexplored regions, but keep it small so the Pareto structure dominates
    exploration_boost = 0.1 * (sigma / np.max(sigma, 1e-8))
    scores = scores + exploration_boost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
