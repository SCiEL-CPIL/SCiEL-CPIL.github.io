"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for the entire evaluation run to enforce deadline
_start_time = None

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline guard: 150s limit, keep 5s margin
    if time.time() - _start_time > 145:
        # Fallback to simple nearest feasible to avoid timeout
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return depot
        return int(feasible[np.argmin(distance_matrix[current_node, feasible])])

    # 1. Identify feasible nodes
    unvisited_array = np.array(unvisited_nodes)
    if len(unvisited_array) == 0:
        return depot
        
    feasible_mask = demands[unvisited_array] <= rest_capacity
    feasible_nodes = unvisited_array[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot

    n_feasible = len(feasible_nodes)
    
    # 2. Regret-based selection with 2-step lookahead
    # For each candidate node j, compute:
    #   - immediate cost: d(current, j)
    #   - regret: how much worse the 2nd-best next step would be if we pick j
    #   - savings: d(current, depot) + d(depot, j) - d(current, j)
    #   - future demand fit: prefer nodes that leave rest_capacity close to a good match
    
    # Immediate distances from current to each feasible node
    dists = distance_matrix[current_node, feasible_nodes]
    
    # Clarke-Wright savings
    d_i_depot = distance_matrix[current_node, depot]
    d_depot_j = distance_matrix[depot, feasible_nodes]
    savings = d_i_depot + d_depot_j - dists
    
    # 2-step lookahead: for each candidate j, what's the best next node after visiting j?
    # This captures whether picking j leads to a good continuation
    # We compute for each j: min over k != j of d(j, k) where k is feasible and demand fits after j
    # To keep it O(n^2) which is fine for n<=50
    
    # Get the sub-matrix of distances between feasible nodes
    sub_dist = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
    
    # For each candidate j, compute the best next step cost (excluding self)
    # After visiting j, remaining capacity is rest_capacity - demands[j]
    # So k is feasible after j if demands[k] <= rest_capacity - demands[j]
    
    best_next_costs = np.zeros(n_feasible)
    second_best_next_costs = np.zeros(n_feasible)
    
    for j_idx in range(n_feasible):
        j_node = feasible_nodes[j_idx]
        remaining_cap_after_j = rest_capacity - demands[j_node]
        
        # Find feasible next nodes after visiting j
        feasible_after_mask = demands[feasible_nodes] <= remaining_cap_after_j
        # Exclude j itself
        feasible_after_mask[j_idx] = False
        
        if not np.any(feasible_after_mask):
            # No next node possible, so the "next cost" is returning to depot
            best_next_costs[j_idx] = distance_matrix[j_node, depot]
            second_best_next_costs[j_idx] = distance_matrix[j_node, depot]
        else:
            next_dists = sub_dist[j_idx, feasible_after_mask]
            # Get sorted distances
            sorted_dists = np.sort(next_dists)
            best_next_costs[j_idx] = sorted_dists[0]
            if len(sorted_dists) > 1:
                second_best_next_costs[j_idx] = sorted_dists[1]
            else:
                second_best_next_costs[j_idx] = sorted_dists[0]
    
    # Regret = second_best - best for the next step
    # High regret means picking j forces a bad next move, so we should pick it now
    regrets = second_best_next_costs - best_next_costs
    
    # 3. Combine scores
    # We want to minimize: immediate_dist + lambda1 * (best_next_cost) - lambda2 * savings + lambda3 * regret_penalty
    # Actually, regret should be MAXIMIZED (pick high-regret nodes first)
    # So score = dists + w1 * best_next_costs - w2 * savings - w3 * regrets
    
    # Tuning: 
    # - w1: how much to look ahead (0.3-0.5 seems reasonable)
    # - w2: savings weight (1.0 is standard for CW)
    # - w3: regret weight (0.5-1.0)
    w1 = 0.4
    w2 = 1.0
    w3 = 0.8
    
    scores = dists + w1 * best_next_costs - w2 * savings - w3 * regrets
    
    # 4. Select best node
    best_idx = np.argmin(scores)
    best_node = feasible_nodes[best_idx]
    
    return int(best_node)
# EVOLVE-BLOCK-END
