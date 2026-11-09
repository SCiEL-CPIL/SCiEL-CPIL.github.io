"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Pareto-frontier filtering and hypervolume contribution scoring.
    
    mu, sigma, f_best, cost: arrays of shape (2000,)
    Returns: array of 2000 finite scores (higher = evaluate next)
    """
    # Deadline guard: if we're close to the 330s limit, return a simple safe result
    if time.time() - _start_time > 310:
        z = (f_best - mu) / (sigma + 1e-9)
        z = np.clip(z, -10, 10)
        EI = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        EI = np.maximum(EI, 0.0)
        scores = EI / (cost + 1e-9)
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against non-finite values
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Compute Expected Improvement (EI) as the improvement objective
    improvement = f_best - mu
    z = improvement / sigma
    z = np.clip(z, -10, 10)
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    ei = np.maximum(ei, 0.0)
    
    # Pareto-frontier filter: retain only candidates that are not dominated
    # in the (improvement, -cost) space. A candidate A dominates B if
    # ei[A] >= ei[B] and cost[A] <= cost[B] (with at least one strict).
    n = len(ei)
    is_pareto = np.ones(n, dtype=bool)
    
    # For each candidate, check if any other candidate dominates it.
    # To be efficient, we can use a sorted approach.
    # Sort by ei descending, then for each candidate, the minimum cost seen so far
    # (among candidates with higher or equal ei) must be less than or equal to its cost
    # for it to be dominated.
    
    # Sort indices by ei descending
    order = np.argsort(-ei, kind='mergesort')
    ei_sorted = ei[order]
    cost_sorted = cost[order]
    
    # Running minimum of cost in the sorted order
    min_cost_so_far = np.full(n, np.inf)
    for i in range(n):
        if i == 0:
            min_cost_so_far[i] = cost_sorted[i]
        else:
            min_cost_so_far[i] = min(min_cost_so_far[i-1], cost_sorted[i])
    
    # A candidate at sorted position i is dominated if there exists a candidate j < i
    # (higher ei) with cost_sorted[j] <= cost_sorted[i].
    # Since we have the running minimum, if min_cost_so_far[i-1] <= cost_sorted[i] (for i>0),
    # then it is dominated. For i=0, it's always on the frontier.
    dominated = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if min_cost_so_far[i-1] <= cost_sorted[i]:
            dominated[i] = True
    
    # Map back to original indices
    dominated_original = np.zeros(n, dtype=bool)
    dominated_original[order] = dominated
    is_pareto = ~dominated_original
    
    # Hypervolume contribution scoring
    # Reference point: (0, max_cost) where max_cost is the maximum cost in the set.
    # Actually, for hypervolume contribution of a point (ei_i, cost_i) relative to
    # the origin (0,0) and the Pareto front, we can approximate its contribution.
    # A simpler and effective approach: score = ei * (max_cost - cost + epsilon)
    # This rewards high improvement and low cost multiplicatively.
    
    max_cost = np.max(cost)
    cost_savings = max_cost - cost + 1e-9
    
    # Score only Pareto-optimal points; others get 0
    scores = np.zeros(n, dtype=np.float64)
    scores[is_pareto] = ei[is_pareto] * cost_savings[is_pareto]
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
