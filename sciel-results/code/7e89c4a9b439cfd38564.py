"""CVRP heuristic: savings-based nearest neighbor with cluster awareness."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level cache
_dm_cache = {}
_start_time = None
_instance_cache = {}

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
    return (time.time() - _start_time) > 140

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Pick the next customer or return to depot (0).
    
    Strategy: Savings-based heuristic with smart cluster formation.
    Core idea: Build routes by connecting nodes that have high mutual savings,
    while keeping the path compact. When no good connections remain, return to depot.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    dm = _get_dm(distance_matrix)
    
    unvisited = list(unvisited_nodes)
    if not unvisited:
        return depot
    
    # Deadline guard
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
    
    # Key distances
    dists_to_current = dm[current_node, feasible_arr]
    dists_to_depot = dm[feasible_arr, depot]
    
    # Sub-matrix of distances among feasible nodes
    sub_dm = dm[np.ix_(feasible_arr, feasible_arr)]
    
    # Savings: S[i,j] = d(depot,i) + d(depot,j) - d(i,j)
    d_depot = dists_to_depot
    savings_matrix = d_depot[:, None] + d_depot[None, :] - sub_dm
    
    # For each node i, compute its "savings potential" with the current node.
    # If we go current -> i -> j, the savings compared to current -> i and current -> j separately
    # is approximately: d(current,j) - d(i,j) + d(depot,i)  (roughly)
    # More precisely, the benefit of going i then j (vs. j then i, or separate trips)
    
    # Primary score: for each candidate i, estimate the cost of a short route starting with i
    # cost_i = d(current, i) + min over j != i of [d(i,j) + d(j, depot)]
    # This captures: go to i, then to the best next node j, then back to depot.
    
    if n > 1:
        # cost_i_j_d[i] = d(current,i) + min_j(d(i,j) + d(j,depot))
        cost_ij_jd = sub_dm + dists_to_depot[None, :]
        np.fill_diagonal(cost_ij_jd, np.inf)
        best_second = np.min(cost_ij_jd, axis=1)
        score_2step = dists_to_current + best_second
    else:
        score_2step = dists_to_current.copy()
    
    # Secondary: 3-step lookahead for better guidance
    # cost_i_j_k_d[i] = d(current,i) + min_j(d(i,j) + min_k(d(j,k) + d(k,depot)))
    if n > 2 and not _check_deadline():
        # Precompute best 2-step from each node to depot
        # best2[j] = min_k (d(j,k) + d(k,depot))
        best2 = np.min(sub_dm + dists_to_depot[None, :], axis=1)
        np.fill_diagonal(best2, 0)  # not quite right, let's redo
        
        # Actually: best2[j] = min over k != j of (d(j,k) + d(k,depot))
        best2 = np.empty(n)
        for j in range(n):
            costs_k = sub_dm[j, :] + dists_to_depot
            costs_k[j] = np.inf
            best2[j] = np.min(costs_k)
        
        # Now for each i: min over j != i of (d(i,j) + best2[j])
        score_3step = np.empty(n)
        for i in range(n):
            costs_j = sub_dm[i, :] + best2
            costs_j[i] = np.inf
            score_3step[i] = dists_to_current[i] + np.min(costs_j)
    else:
        score_3step = score_2step
    
    # Savings-based bonus: nodes that have high savings with other feasible nodes
    # should be visited earlier to form tight clusters
    total_savings = np.sum(savings_matrix, axis=1)
    
    # Also consider: savings with the current node specifically
    # If we go current -> i, the "savings" relative to going current -> depot -> i
    # is: d(current, depot) + d(depot, i) - d(current, i)
    # This is the classic savings for inserting i after current
    savings_with_current = dm[current_node, depot] + dists_to_depot - dists_to_current
    
    # Capacity-aware adjustment
    demands_arr = np.array([demands[i] for i in feasible], dtype=np.float64)
    total_remaining_demand = np.sum(demands_arr)
    
    # If capacity is tight, prefer heavier nodes (they're harder to place later)
    if total_remaining_demand > 0:
        cap_util = rest_capacity / total_remaining_demand
    else:
        cap_util = 1.0
    
    # Normalize scores for weighting
    max_s2 = np.max(score_2step) if np.max(score_2step) > 0 else 1.0
    max_s3 = np.max(score_3step) if np.max(score_3step) > 0 else 1.0
    max_sv = np.max(total_savings) if np.max(total_savings) > 0 else 1.0
    max_swc = np.max(savings_with_current) if np.max(savings_with_current) > 0 else 1.0
    
    norm_s2 = score_2step / max_s2
    norm_s3 = score_3step / max_s3
    norm_sv = total_savings / max_sv
    norm_swc = savings_with_current / max_swc
    
    # Adaptive weights based on how many nodes remain
    if n > 25:
        # Early: emphasize cluster formation (savings) and 3-step lookahead
        w_s2, w_s3, w_sv, w_swc, w_cap = 0.2, 0.4, 0.25, 0.1, 0.05
    elif n > 15:
        w_s2, w_s3, w_sv, w_swc, w_cap = 0.3, 0.3, 0.25, 0.1, 0.05
    elif n > 8:
        w_s2, w_s3, w_sv, w_swc, w_cap = 0.4, 0.25, 0.2, 0.1, 0.05
    else:
        # Late: emphasize immediate cost and return to depot
        w_s2, w_s3, w_sv, w_swc, w_cap = 0.5, 0.2, 0.15, 0.1, 0.05
    
    # Capacity term
    max_demand = np.max(demands_arr) if np.max(demands_arr) > 0 else 1.0
    norm_demand = demands_arr / max_demand
    
    if cap_util < 0.3:
        capacity_term = norm_demand
    elif cap_util > 0.7:
        capacity_term = 1.0 - norm_demand
    else:
        capacity_term = 0.5 * np.ones(n)
    
    # Composite score (lower is better)
    # We want to minimize routing cost, so we combine:
    # - Low 2-step and 3-step costs (good)
    # - High savings (good, so subtract)
    # - High savings with current (good, so subtract)
    scores = (w_s2 * norm_s2 + w_s3 * norm_s3 
              - w_sv * norm_sv - w_swc * norm_swc 
              + w_cap * capacity_term)
    
    # Additional tiebreaker: if scores are very close, prefer the one with higher savings
    min_score = np.min(scores)
    close_mask = scores < min_score + 0.01
    if np.sum(close_mask) > 1:
        # Among close candidates, pick the one with highest total savings
        close_indices = np.where(close_mask)[0]
        best_close = close_indices[np.argmax(total_savings[close_indices])]
        best_idx = best_close
    else:
        best_idx = np.argmin(scores)
    
    return int(feasible_arr[best_idx])
# EVOLVE-BLOCK-END
