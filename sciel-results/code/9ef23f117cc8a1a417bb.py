"""Cost-aware acquisition for Ackley 2D BO with Pareto-frontier hypervolume scoring."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0

def _check_deadline():
    return time.time() > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    if _check_deadline():
        mu_safe = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
        cost_safe = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e6)
        with np.errstate(divide='ignore', invalid='ignore'):
            fallback_scores = -mu_safe / (cost_safe + 1e-9)
        return np.nan_to_num(fallback_scores, nan=0.0, posinf=0.0, neginf=0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=0.0)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e6)

    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-6)

    # Compute expected improvement (EI) as the "improvement" objective
    # EI = E[max(f_best - f(x), 0)] under GP posterior
    z = (f_best - mu) / sigma
    from scipy.stats import norm
    ei = sigma * (z * norm.cdf(z) + norm.pdf(z))
    # Clip small negative values from numerical error
    ei = np.maximum(ei, 0.0)

    # Multi-objective: maximize improvement (ei), minimize cost
    # Pareto filter: keep only non-dominated points
    # A point dominates another if it has >= ei AND <= cost, with at least one strict
    n = len(ei)
    is_pareto = np.ones(n, dtype=bool)
    
    # Efficient Pareto front computation
    # Sort by ei descending, then by cost ascending
    order = np.lexsort((cost, -ei))
    ei_sorted = ei[order]
    cost_sorted = cost[order]
    
    # A point is on the Pareto front if no previously seen point has both
    # higher/equal ei and lower/equal cost (with at least one strict)
    # Since sorted by ei descending, we track the minimum cost seen so far
    min_cost_so_far = np.inf
    pareto_mask = np.zeros(n, dtype=bool)
    for i in range(n):
        if cost_sorted[i] < min_cost_so_far:
            pareto_mask[order[i]] = True
            min_cost_so_far = cost_sorted[i]
    
    # For points not on the Pareto front, give them a low score
    # For points on the Pareto front, compute hypervolume contribution
    
    # Hypervolume contribution: for each Pareto point, the area it uniquely contributes
    # relative to the reference point (0, max_cost)
    # In 2D, hypervolume contribution of point (ei_i, cost_i) with reference (0, C_ref)
    # is the rectangle from (ei_i, cost_i) to (next_point_ei_or_max, C_ref)
    
    C_ref = float(np.max(cost)) + 1e-6
    
    scores = np.zeros(n, dtype=np.float64)
    
    if np.any(pareto_mask):
        # Get Pareto points sorted by ei descending
        pareto_idx = np.where(pareto_mask)[0]
        pareto_ei = ei[pareto_idx]
        pareto_cost = cost[pareto_idx]
        
        # Sort by ei descending
        p_order = np.argsort(-pareto_ei)
        pareto_ei_sorted = pareto_ei[p_order]
        pareto_cost_sorted = pareto_cost[p_order]
        pareto_idx_sorted = pareto_idx[p_order]
        
        # Compute hypervolume contribution for each point
        # For the i-th point (sorted by ei desc), its contribution is:
        # (ei_i - ei_{i+1}) * (C_ref - cost_i) for i < last
        # For the last point: ei_last * (C_ref - cost_last)
        n_pareto = len(pareto_idx_sorted)
        for i in range(n_pareto):
            if i < n_pareto - 1:
                ei_width = pareto_ei_sorted[i] - pareto_ei_sorted[i + 1]
            else:
                ei_width = pareto_ei_sorted[i]
            
            cost_height = C_ref - pareto_cost_sorted[i]
            
            hv_contrib = ei_width * cost_height
            scores[pareto_idx_sorted[i]] = hv_contrib
    
    # Add a small exploration bonus based on sigma to break ties and encourage diversity
    # among Pareto-optimal points
    exploration_bonus = 0.1 * sigma
    
    # Combine: hypervolume contribution + exploration bonus
    scores = scores + exploration_bonus
    
    # Ensure dominated points get very low scores
    scores[~pareto_mask] = np.minimum(scores[~pareto_mask], 1e-9)
    
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    return scores
# EVOLVE-BLOCK-END
