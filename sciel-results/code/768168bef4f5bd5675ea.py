"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
    
    # Safety margin: 150s limit, stop at 145s
    if time.time() - _start_time > 145:
        return int(depot)
    
    if len(unvisited_nodes) == 0:
        return int(depot)
    
    unvisited_arr = np.array(unvisited_nodes, dtype=int)
    n_unvisited = len(unvisited_arr)
    
    # Filter feasible nodes by capacity
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    feasible_nodes = unvisited_arr[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return int(depot)
    
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])

    # Pre-compute distances from current node to feasible nodes
    d_current_to_feasible = distance_matrix[current_node, feasible_nodes]
    
    # Pre-compute distances from feasible nodes to depot
    d_feasible_to_depot = distance_matrix[feasible_nodes, depot]
    
    # Pre-compute demands of feasible nodes
    d_feasible_demands = demands[feasible_nodes]
    
    # We will compute a 2-step lookahead score for each feasible node.
    # Score(i) = d(current, i) + SoftMin_{j in Feasible(i)} (d(i, j) + d(j, depot))
    # If no j is feasible, SoftMin term is d(i, depot).
    
    scores = np.zeros(len(feasible_nodes))
    
    # Temperature for soft-min (log-sum-exp)
    # Smaller tau -> closer to hard min. 
    # Distances are likely in range 0-100. tau=1.0 is a reasonable smoothness.
    tau = 1.0
    
    for idx_i in range(len(feasible_nodes)):
        i = feasible_nodes[idx_i]
        d_c_i = d_current_to_feasible[idx_i]
        d_i_depot = d_feasible_to_depot[idx_i]
        
        rem_cap_after_i = rest_capacity - d_feasible_demands[idx_i]
        
        # Find feasible j's for the second step
        # j must be in feasible_nodes, j != i, and demand(j) <= rem_cap_after_i
        if rem_cap_after_i <= 0:
            # No second step possible
            scores[idx_i] = d_c_i + d_i_depot
            continue
        
        # Get indices of feasible nodes that can follow i
        # We need to check demand of all feasible nodes against rem_cap_after_i
        # and exclude i itself.
        
        # Create a mask for j's
        # demands[feasible_nodes] is d_feasible_demands
        # We need j != i, so we can't just use index directly if we want to be careful, 
        # but since node IDs are unique, we can compare IDs or indices.
        
        # Let's use indices in feasible_nodes
        # d_feasible_demands[k] is demand of feasible_nodes[k]
        
        # Mask for capacity
        cap_mask = d_feasible_demands <= rem_cap_after_i
        
        # Mask for not being i
        # feasible_nodes[idx_i] is the node ID
        not_i_mask = feasible_nodes != i
        
        # Combined mask
        valid_j_mask = cap_mask & not_i_mask
        
        if not np.any(valid_j_mask):
            # No valid second step
            scores[idx_i] = d_c_i + d_i_depot
            continue
        
        # Get the valid j's
        valid_js = feasible_nodes[valid_j_mask]
        
        # Calculate cost for each valid j: d(i, j) + d(j, depot)
        d_i_to_js = distance_matrix[i, valid_js]
        d_js_to_depot = distance_matrix[valid_js, depot]
        
        costs_j = d_i_to_js + d_js_to_depot
        
        # Apply soft-min
        # SoftMin(x) = -tau * log(sum(exp(-x/tau)))
        # To avoid overflow/underflow, subtract min
        min_cost_j = np.min(costs_j)
        exp_terms = np.exp(-(costs_j - min_cost_j) / tau)
        soft_min_val = min_cost_j - tau * np.log(np.sum(exp_terms))
        
        scores[idx_i] = d_c_i + soft_min_val
        
    # Select the best node
    best_idx = int(np.argmin(scores))
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
