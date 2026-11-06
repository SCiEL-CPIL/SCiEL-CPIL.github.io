"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
    
    # Safety margin: 120s limit, stop at 118s
    if time.time() - _start_time > 118:
        return int(depot)
    
    if len(unvisited_nodes) == 0:
        return int(depot)
        
    unvisited_arr = np.array(unvisited_nodes, dtype=int)
    
    # Filter feasible nodes by capacity
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    feasible_nodes = unvisited_arr[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return int(depot)
        
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])
        
    best_score = float('inf')
    best_node = -1
    
    # Pre-compute distance from current to depot (constant for this step)
    d_current_depot = distance_matrix[current_node, depot]
    
    # Vectorized distance from current to all feasible nodes
    d_current_feasible = distance_matrix[current_node, feasible_nodes]
    
    # Soft-min temperature for smooth lookahead
    tau = 0.1
    
    # Pre-compute distances from depot to all feasible nodes for savings
    d_depot_feasible = distance_matrix[depot, feasible_nodes]
    
    # Loop over feasible nodes to calculate lookahead and savings
    for idx, j in enumerate(feasible_nodes):
        d_ij = d_current_feasible[idx]
        
        # Calculate savings: d(i, depot) + d(depot, j) - d(i, j)
        savings = d_current_depot + d_depot_feasible[idx] - d_ij
        
        # Calculate soft-min lookahead: -log(sum(exp(-d/tau)))/log(n)
        remaining_cap = rest_capacity - demands[j]
        
        # Find nodes that can be visited after j
        # They must be in unvisited, not j, and demand <= remaining_cap
        cap_mask = demands[unvisited_arr] <= remaining_cap
        not_j_mask = unvisited_arr != j
        
        next_feasible_mask = cap_mask & not_j_mask
        next_feasible_nodes = unvisited_arr[next_feasible_mask]
        
        # Always include depot as a candidate for the next step
        # But we need to be careful: if we go to depot, route closes
        # So we should consider both: min over (next_feasible_nodes, depot)
        # Actually, the original code treated them separately.
        # Let's combine: candidates are next_feasible_nodes + [depot]
        
        if len(next_feasible_nodes) > 0:
            # Get distances from j to all next feasible nodes
            d_to_next = distance_matrix[j, next_feasible_nodes]
            # Add depot distance
            d_to_depot = distance_matrix[j, depot]
            # Combine
            all_next_dists = np.concatenate([d_to_next, [d_to_depot]])
        else:
            all_next_dists = np.array([distance_matrix[j, depot]])
        
        # Soft-min: -tau * log(sum(exp(-d/tau)))
        # This is a smooth approximation of min(d)
        # Using log-sum-exp trick for numerical stability
        min_d = np.min(all_next_dists)
        exp_terms = np.exp(-(all_next_dists - min_d) / tau)
        soft_min = min_d - tau * np.log(np.sum(exp_terms))
        
        # Score: Minimize immediate + lookahead, maximize savings
        # Add a small regret term to prefer nodes that are much better than second-best
        # Regret = (second_best_score - best_score) for this node's position
        # But we don't have other scores yet. 
        # Alternative: use a simple penalty for high demand to prioritize capacity usage?
        # Let's stick with the refined soft-min approach first.
        
        score = d_ij + soft_min - 1.0 * savings
        
        if score < best_score:
            best_score = score
            best_node = j
            
    return int(best_node)
# EVOLVE-BLOCK-END
