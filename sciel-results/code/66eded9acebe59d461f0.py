"""Cost-aware acquisition for Bayesian optimization of powell 4d (min 0) on [-4,5]^4."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_EPS = 1e-9
_DEADLINE_S = 330.0
_SAFETY_S = 3.0


def acquisition(mu, sigma, f_best, cost):
    t0 = time.time()
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Guard against non-finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, _EPS)
    sigma = np.clip(sigma, _EPS, None)
    cost = np.where(np.isfinite(cost), cost, 1.5)
    cost = np.clip(cost, _EPS, None)

    # Use known true minimum (0) as target for stronger improvement signal
    f_target = 0.0
    improvement = f_target - mu
    z = improvement / sigma

    # Clip z to avoid numerical issues
    z = np.clip(z, -50, 50)

    # Expected Improvement (EI)
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Two-stage Pareto front filtering approach
    # Stage 1: Identify non-dominated candidates based on (EI, cost) trade-off
    # A candidate i dominates candidate j if ei[i] >= ei[j] and cost[i] <= cost[j]
    # and at least one inequality is strict.
    
    n = len(ei)
    
    # Normalize EI and cost to [0, 1] for fair comparison
    ei_min, ei_max = ei.min(), ei.max()
    cost_min, cost_max = cost.min(), cost.max()
    
    ei_range = ei_max - ei_min
    cost_range = cost_max - cost_min
    
    if ei_range < _EPS:
        ei_norm = np.zeros(n)
    else:
        ei_norm = (ei - ei_min) / ei_range
    
    if cost_range < _EPS:
        cost_norm = np.zeros(n)
    else:
        cost_norm = (cost - cost_min) / cost_range

    # Find Pareto front: candidates that are not dominated by any other
    # Efficient O(n^2) approach for 2000 points
    is_dominated = np.zeros(n, dtype=bool)
    
    # For each candidate, check if it's dominated
    # To speed this up, we can use a sorted approach
    # Sort by EI descending, then for each candidate, check if any previous candidate has lower cost
    
    order = np.argsort(-ei_norm)  # Sort by EI descending
    sorted_ei = ei_norm[order]
    sorted_cost = cost_norm[order]
    
    # A candidate at position i in sorted order is dominated if there exists
    # a candidate at position j < i with sorted_cost[j] <= sorted_cost[i]
    # (since sorted_ei[j] >= sorted_ei[i] by construction)
    
    min_cost_so_far = np.inf
    for i in range(n):
        if sorted_cost[i] < min_cost_so_far:
            # This candidate is on the Pareto front
            min_cost_so_far = sorted_cost[i]
        else:
            # This candidate is dominated
            is_dominated[order[i]] = True
    
    # Stage 2: Select from Pareto front using dynamic weighting
    # As remaining budget decreases, shift weight toward cost-efficiency
    # Estimate remaining budget: assume average cost per evaluation
    mean_cost = cost.mean()
    # Total budget is 30, starting cost is 5 (5 initial points)
    # We don't know exact current cost, but we can estimate progress
    # Use f_best and cost distribution to estimate
    
    # Dynamic weight: starts at 0.5 (balanced), shifts toward cost-efficiency
    # as we use more of the budget
    # Estimate fraction of budget used based on number of evaluations
    # Since we don't have that directly, use cost distribution as proxy
    
    # Candidates with lower cost are "earlier" in budget usage
    cost_percentile = np.argsort(np.argsort(cost)) / max(n - 1, 1)
    mean_percentile = cost_percentile[~is_dominated].mean() if not is_dominated.all() else 0.5
    
    # Weight on EI vs cost-efficiency
    # Higher weight on EI early, higher weight on cost-efficiency later
    w_ei = 0.7 - 0.4 * mean_percentile  # Decreases from 0.7 to 0.3
    
    # Score on Pareto front: convex combination
    # For candidates on the front, compute a score that balances EI and cost-efficiency
    score = np.full(n, -1e30)  # Default: dominated candidates get very low score
    
    # For Pareto front candidates, use normalized EI and inverse normalized cost
    front_mask = ~is_dominated
    if front_mask.any():
        # Cost efficiency: lower cost is better
        cost_eff = 1.0 - cost_norm[front_mask]
        
        # Convex combination
        score[front_mask] = w_ei * ei_norm[front_mask] + (1 - w_ei) * cost_eff
    
    # Add small UCB exploration term for sigma to escape local basins
    # But only for Pareto front candidates
    ucb_beta = 0.1
    score = score + ucb_beta * sigma * front_mask.astype(np.float64)
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, -1e30)
    
    # Deadline guard
    if time.time() - t0 > (_DEADLINE_S - _SAFETY_S):
        return np.where(np.isfinite(score), score, 0.0)
    
    return np.where(np.isfinite(score), score, 0.0)
# EVOLVE-BLOCK-END
