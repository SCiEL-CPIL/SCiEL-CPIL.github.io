"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level cache
_dm_cache = {}
_start_time = None
_total_unvisited_initial = None

def _get_dm(distance_matrix):
    """Get or create numpy view of distance matrix."""
    key = id(distance_matrix)
    if key not in _dm_cache:
        _dm_cache[key] = np.asarray(distance_matrix, dtype=np.float64)
    return _dm_cache[key]

def _check_deadline():
    """Return True if we are close to the time limit."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    # Safety margin of 5 seconds
    return (time.time() - _start_time) > 145

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Pick the next customer or return to depot (0).
    
    Strategy: Adaptive Savings + 2-Step Lookahead.
    - Raw savings-based scoring to build compact clusters.
    - 2-step lookahead to anticipate next move cost.
    - Phase-adaptive weights: more savings early, more depot-return late.
    """
    global _start_time, _total_unvisited_initial
    if _start_time is None:
        _start_time = time.time()
        _total_unvisited_initial = len(unvisited_nodes)
        
    dm = _get_dm(distance_matrix)
    
    unvisited = list(unvisited_nodes)
    if not unvisited:
        return depot
    
    # Deadline guard: fall back to nearest feasible
    if _check_deadline():
        feasible = [n for n in unvisited if demands[n] <= rest_capacity]
        if not feasible:
            return depot
        if len(feasible) == 1:
            return int(feasible[0])
        feasible_arr = np.array(feasible)
        dists = dm[current_node, feasible_arr]
        best_idx = np.argmin(dists)
        return int(feasible_arr[best_idx])
    
    # Filter to capacity-feasible nodes
    feasible = [n for n in unvisited if demands[n] <= rest_capacity]
    
    if not feasible:
        return depot
    
    if len(feasible) == 1:
        return int(feasible[0])
    
    feasible_arr = np.array(feasible, dtype=np.int64)
    n = len(feasible_arr)
    
    # Compute distances
    dists_to_current = dm[current_node, feasible_arr]
    dists_to_depot = dm[feasible_arr, depot]
    
    # Sub-matrix of distances among feasible nodes
    sub_dm = dm[np.ix_(feasible_arr, feasible_arr)].copy()
    
    # Savings: S[i,j] = d(depot,i) + d(depot,j) - d(i,j)
    # Total savings for each node with all others
    savings_matrix = dists_to_depot[:, None] + dists_to_depot[None, :] - sub_dm
    # Set diagonal to 0 (no self-savings)
    np.fill_diagonal(savings_matrix, 0.0)
    total_savings = np.sum(savings_matrix, axis=1)
    
    # 2-step lookahead: for each candidate i, find best j != i minimizing d(i,j) + d(j, depot)
    cost_ij_jd = sub_dm + dists_to_depot[None, :]
    np.fill_diagonal(cost_ij_jd, np.inf)
    best_second_step = np.min(cost_ij_jd, axis=1)
    
    # Phase factor: fraction of nodes remaining
    if _total_unvisited_initial and _total_unvisited_initial > 0:
        phase = n / float(_total_unvisited_initial)  # 1.0 at start, ~0 at end
    else:
        phase = 1.0
    
    # Adaptive weights:
    # Early (phase ~ 1): weight savings heavily, less lookahead
    # Late (phase ~ 0): weight depot distance, more lookahead
    alpha = 0.3 + 0.4 * phase    # savings weight: 0.3-0.7
    beta = 0.1 + 0.3 * (1 - phase)  # lookahead weight: 0.1-0.4
    gamma = 0.05 + 0.15 * (1 - phase)  # depot distance weight: 0.05-0.2
    
    # Composite score (minimize):
    # - dists_to_current: immediate cost (weight 1.0, implicit)
    # - alpha * total_savings: reward high savings (subtract)
    # - beta * best_second_step: penalize bad 2nd step
    # - gamma * dists_to_depot: penalize far-from-depot (especially late)
    scores = dists_to_current - alpha * total_savings + beta * best_second_step + gamma * dists_to_depot
    
    # Pick the feasible node with the lowest score
    best_idx = np.argmin(scores)
    return int(feasible_arr[best_idx])
# EVOLVE-BLOCK-END
