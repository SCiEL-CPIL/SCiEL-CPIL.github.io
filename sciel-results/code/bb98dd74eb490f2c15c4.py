"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _time_exceeded():
    return (time.time() - _START_TIME) > _TIME_LIMIT

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node using a 2-step lookahead heuristic.
    Returns 0 (depot) if no feasible nodes or to close route.
    """
    if len(unvisited_nodes) == 0:
        return depot

    # Convert to numpy array for fast operations
    unvisited_arr = np.array(unvisited_nodes, dtype=int)
    
    # 1. Filter feasible nodes based on capacity
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    feasible_nodes = unvisited_arr[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot

    # If time is running out, fallback to simple nearest neighbor
    if _time_exceeded():
        dists = distance_matrix[current_node, feasible_nodes]
        return int(feasible_nodes[np.argmin(dists)])

    # 2. Calculate scores for each feasible node
    # We want to minimize: d(current, j) + min(d(j, k) for k in feasible_next(j)) 
    # If no k exists, min cost is d(j, depot)
    
    best_score = np.inf
    best_node = feasible_nodes[0]
    
    # Pre-compute distances from current to all feasible nodes
    dist_current_to_feasible = distance_matrix[current_node, feasible_nodes]
    
    for idx, j in enumerate(feasible_nodes):
        cost_to_j = dist_current_to_feasible[idx]
        
        # Find best next node k from j
        # Feasible next nodes must satisfy: demand[k] <= rest_capacity - demand[j]
        remaining_cap_after_j = rest_capacity - demands[j]
        
        # Candidates for k are other feasible nodes (excluding j)
        # Note: We can use the original unvisited_nodes list for k, 
        # but must ensure k is not j and demand[k] fits.
        
        # Optimization: Instead of looping all unvisited, we can just check the feasible_nodes list
        # but we need to exclude j.
        
        min_next_cost = np.inf
        # Check all other feasible nodes as potential next steps
        # This is O(N^2) in the worst case per step, but N is small (<=50)
        
        # Get demands of all feasible nodes
        # We already have feasible_nodes.
        # We need to check which of them can follow j.
        
        # Vectorized check:
        # demands[feasible_nodes] <= remaining_cap_after_j
        # AND feasible_nodes != j
        
        if len(feasible_nodes) > 1:
            # Mask for nodes that can follow j
            can_follow_mask = (demands[feasible_nodes] <= remaining_cap_after_j) & (feasible_nodes != j)
            
            if np.any(can_follow_mask):
                potential_nexts = feasible_nodes[can_follow_mask]
                # Get distances from j to these potential nexts
                dists_j_to_nexts = distance_matrix[j, potential_nexts]
                min_next_cost = np.min(dists_j_to_nexts)
        
        if np.isinf(min_next_cost):
            # No feasible next node, so we must return to depot
            min_next_cost = distance_matrix[j, depot]
            
        total_score = cost_to_j + min_next_cost
        
        if total_score < best_score:
            best_score = total_score
            best_node = j

    return int(best_node)
# EVOLVE-BLOCK-END
