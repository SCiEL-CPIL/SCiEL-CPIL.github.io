"""Model-and-optimize approach for capacitated vehicle routing."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

# Module-level state
_dm_cache = {}
_start_time = None
_instance_id = 0

def _get_dm(distance_matrix):
    """Get or create numpy view of distance matrix."""
    global _instance_id
    key = id(distance_matrix)
    if key not in _dm_cache:
        _dm_cache[key] = np.asarray(distance_matrix, dtype=np.float64)
        _instance_id += 1
        _start_time = time.time()
    return _dm_cache[key]

def _check_deadline():
    """Return True if we are close to the time limit."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return (time.time() - _start_time) > 140

def _greedy_complete_route(start_node, depot, unvisited, rest_capacity, demands, dm):
    """
    Fast greedy heuristic to complete a route from start_node.
    Returns the total distance of the completed route (from start_node to depot).
    """
    total_dist = 0.0
    current = start_node
    remaining_cap = rest_capacity
    remaining = list(unvisited)
    
    while remaining:
        feasible = [n for n in remaining if demands[n] <= remaining_cap]
        if not feasible:
            break
        
        if len(feasible) == 1:
            next_node = feasible[0]
        else:
            feas_arr = np.array(feasible)
            dists_to_current = dm[current, feas_arr]
            dists_to_depot = dm[feas_arr, depot]
            scores = dists_to_current + 0.1 * dists_to_depot
            best_idx = np.argmin(scores)
            next_node = int(feas_arr[best_idx])
        
        total_dist += dm[current, next_node]
        current = next_node
        remaining_cap -= demands[next_node]
        remaining.remove(next_node)
    
    total_dist += dm[current, depot]
    return total_dist

def _improved_greedy_complete_route(start_node, depot, unvisited, rest_capacity, demands, dm):
    """
    Improved greedy that looks ahead 1-2 steps.
    """
    total_dist = 0.0
    current = start_node
    remaining_cap = rest_capacity
    remaining = list(unvisited)
    
    while remaining:
        feasible = [n for n in remaining if demands[n] <= remaining_cap]
        if not feasible:
            break
        
        if len(feasible) == 1:
            next_node = feasible[0]
        elif len(feasible) <= 6:
            best_cost = float('inf')
            best_node = feasible[0]
            for cand in feasible:
                sim_remaining = [n for n in remaining if n != cand]
                sim_cap = remaining_cap - demands[cand]
                if sim_cap < 0:
                    continue
                cost = dm[current, cand] + _greedy_complete_route(cand, depot, sim_remaining, sim_cap, demands, dm)
                if cost < best_cost:
                    best_cost = cost
                    best_node = cand
            next_node = best_node
        else:
            feas_arr = np.array(feasible)
            dists_to_current = dm[current, feas_arr]
            dists_to_depot = dm[feas_arr, depot]
            scores = dists_to_current + 0.15 * dists_to_depot
            best_idx = np.argmin(scores)
            next_node = int(feas_arr[best_idx])
        
        total_dist += dm[current, next_node]
        current = next_node
        remaining_cap -= demands[next_node]
        remaining.remove(next_node)
    
    total_dist += dm[current, depot]
    return total_dist

def _combined_score_select(current_node, depot, unvisited, rest_capacity, demands, dm):
    """
    Combined density-centroid and savings-based selection.
    Computes both a demand-weighted centroid centrality score and a Clarke-Wright
    savings score, then combines them with a soft-min (log-sum-exp) to get a smooth
    surrogate for the best of the two scoring functions.
    """
    feasible = [n for n in unvisited if demands[n] <= rest_capacity]
    if not feasible:
        return depot
    if len(feasible) == 1:
        return int(feasible[0])
    
    feas_arr = np.array(feasible)
    n_feasible = len(feas_arr)
    
    # Demand weights for density computation
    demand_weights = np.array([demands[n] for n in feas_arr], dtype=np.float64)
    total_demand = demand_weights.sum()
    if total_demand < 1e-9:
        demand_weights = np.ones_like(demand_weights)
        total_demand = demand_weights.sum()
    demand_weights = demand_weights / total_demand
    
    # Compute density-weighted centrality
    dm_sub = dm[np.ix_(feas_arr, feas_arr)]
    centrality = dm_sub @ demand_weights
    
    c_min = np.min(centrality)
    c_max = np.max(centrality)
    if c_max > c_min:
        norm_central = (centrality - c_min) / (c_max - c_min)
    else:
        norm_central = np.zeros_like(centrality)
    
    scale_c = c_max - c_min if c_max > c_min else 1.0
    centrality_scaled = norm_central * scale_c
    
    # Distances
    dist_current = dm[current_node, feas_arr]
    dist_depot = dm[feas_arr, depot]
    dist_current_depot = dm[current_node, depot]
    
    # Savings: d(current,depot) + d(depot,cand) - d(current,cand)
    savings = dist_current_depot + dist_depot - dist_current
    
    s_min = np.min(savings)
    s_max = np.max(savings)
    if s_max > s_min:
        norm_savings = (savings - s_min) / (s_max - s_min)
    else:
        norm_savings = np.zeros_like(savings)
    
    # Score 1: density-centroid based
    score_density = dist_current + 0.3 * centrality_scaled + 0.1 * dist_depot
    
    # Score 2: savings-based
    score_savings = dist_current + 0.1 * dist_depot - 0.5 * savings
    
    # Soft-min via log-sum-exp (temperature controls smoothness)
    # Lower score is better, so we want min(score_density, score_savings)
    # soft_min = -logsumexp(-[a,b]/T) * T
    T = 5.0  # temperature parameter
    neg_scores = np.stack([-score_density, -score_savings], axis=1) / T
    # log-sum-exp for numerical stability
    max_neg = np.max(neg_scores, axis=1)
    logsumexp_vals = max_neg + np.log(np.sum(np.exp(neg_scores - max_neg[:, np.newaxis]), axis=1))
    soft_min_score = -logsumexp_vals * T
    
    best_idx = np.argmin(soft_min_score)
    return int(feas_arr[best_idx])

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Pick the next customer or return to depot (0).
    
    Strategy: For each feasible candidate, estimate the total cost of the
    remaining route if we pick that candidate. Choose the candidate that
    minimizes this estimate. Uses a combined density-centroid + savings
    scoring with soft-min surrogate to filter top-k candidates before
    doing lookahead.
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
    
    n_feasible = len(feasible)
    
    # If few feasible nodes, do full lookahead
    if n_feasible <= 6:
        best_cost = float('inf')
        best_node = feasible[0]
        
        for cand in feasible:
            sim_remaining = [n for n in unvisited if n != cand]
            sim_cap = rest_capacity - demands[cand]
            if sim_cap < 0:
                continue
            sim_cost = _improved_greedy_complete_route(cand, depot, sim_remaining, sim_cap, demands, dm)
            total_cost = dm[current_node, cand] + sim_cost
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_node = cand
        
        return int(best_node)
    else:
        # Hybrid: use combined density+savings scoring to filter top-k, then lookahead
        feas_arr = np.array(feasible)
        
        # Compute combined score for filtering
        demand_weights = np.array([demands[n] for n in feas_arr], dtype=np.float64)
        total_demand = demand_weights.sum()
        if total_demand < 1e-9:
            demand_weights = np.ones_like(demand_weights)
            total_demand = demand_weights.sum()
        demand_weights = demand_weights / total_demand
        
        dm_sub = dm[np.ix_(feas_arr, feas_arr)]
        centrality = dm_sub @ demand_weights
        
        c_min = np.min(centrality)
        c_max = np.max(centrality)
        if c_max > c_min:
            norm_central = (centrality - c_min) / (c_max - c_min)
        else:
            norm_central = np.zeros_like(centrality)
        
        scale_c = c_max - c_min if c_max > c_min else 1.0
        centrality_scaled = norm_central * scale_c
        
        dists = dm[current_node, feas_arr]
        dists_depot = dm[feas_arr, depot]
        dist_current_depot = dm[current_node, depot]
        
        savings = dist_current_depot + dists_depot - dists
        
        s_min = np.min(savings)
        s_max = np.max(savings)
        if s_max > s_min:
            norm_savings = (savings - s_min) / (s_max - s_min)
        else:
            norm_savings = np.zeros_like(savings)
        
        score_density = dists + 0.3 * centrality_scaled + 0.1 * dists_depot
        score_savings = dists + 0.1 * dists_depot - 0.5 * savings
        
        # Soft-min via log-sum-exp
        T = 5.0
        neg_scores = np.stack([-score_density, -score_savings], axis=1) / T
        max_neg = np.max(neg_scores, axis=1)
        logsumexp_vals = max_neg + np.log(np.sum(np.exp(neg_scores - max_neg[:, np.newaxis]), axis=1))
        soft_min_score = -logsumexp_vals * T
        
        # Get top-k by combined soft-min score
        k = min(6, n_feasible)
        top_k_indices = np.argpartition(soft_min_score, k - 1)[:k]
        top_k_nodes = [int(feas_arr[i]) for i in top_k_indices]
        
        best_cost = float('inf')
        best_node = top_k_nodes[0]
        
        for cand in top_k_nodes:
            sim_remaining = [n for n in unvisited if n != cand]
            sim_cap = rest_capacity - demands[cand]
            if sim_cap < 0:
                continue
            sim_cost = _improved_greedy_complete_route(cand, depot, sim_remaining, sim_cap, demands, dm)
            total_cost = dm[current_node, cand] + sim_cost
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_node = cand
        
        return int(best_node)
# EVOLVE-BLOCK-END
