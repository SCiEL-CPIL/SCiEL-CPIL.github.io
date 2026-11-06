"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Local Regret Heuristic: For each feasible customer, compute the difference
    between the distance to its nearest feasible alternative (other customer)
    and the distance to its second-nearest feasible alternative.
    
    This prioritizes nodes that are structurally "isolated" (large gap between
    closest and second-closest neighbors), suggesting they should be visited
    early to secure their unique connection, or nodes in dense clusters depending
    on the sign. We use MAX regret (isolation) here to prevent leaving isolated
    nodes for last which would cause long detours.
    """
    if len(unvisited_nodes) == 0:
        return depot

    # Filter to only feasible nodes (demand fits in remaining capacity)
    feasible_mask = demands[unvisited_nodes] <= rest_capacity
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        # No feasible customers left, go back to depot
        return depot

    # Get the feasible unvisited node indices (original node IDs)
    feasible_nodes = unvisited_nodes[feasible_indices]
    
    # If only one feasible node, go there
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])
        
    # Fallback to Nearest Neighbor if too few nodes to calculate robust regret
    if len(feasible_nodes) < 3:
        dists = distance_matrix[current_node, feasible_nodes]
        return int(feasible_nodes[np.argmin(dists)])

    # Calculate local regret for each feasible node
    # Regret_i = d(i, 2nd_nearest_other) - d(i, nearest_other)
    regrets = np.zeros(len(feasible_nodes))
    
    for idx, node in enumerate(feasible_nodes):
        # Get indices of all other feasible nodes
        other_indices = np.concatenate([np.arange(idx), np.arange(idx+1, len(feasible_nodes))])
        other_nodes = feasible_nodes[other_indices]
        
        # Distances from 'node' to 'other_nodes'
        dists = distance_matrix[node, other_nodes]
        
        # Find the two smallest distances
        # Using partition for efficiency, though N is small
        if len(dists) >= 2:
            # Get the two smallest values
            # np.partition returns the k-th smallest at index k-1? No, index k is the (k+1)-th smallest?
            # np.partition(a, k) -> a[k] is the (k+1)-th smallest element.
            # We want the 1st and 2nd smallest.
            # So k=1. The element at index 1 is the 2nd smallest. Elements at 0..1 are <= a[1].
            # So a[:2] contains the two smallest, but not sorted.
            two_smallest = np.partition(dists, 1)[:2]
            d1 = np.min(two_smallest)
            d2 = np.max(two_smallest)
            regrets[idx] = d2 - d1
        else:
            # Should not happen due to check above
            regrets[idx] = 0.0
            
    # Select node with maximum regret (most isolated/unique connection)
    max_regret = np.max(regrets)
    
    # Handle ties: pick the one closest to current_node
    candidates_mask = regrets == max_regret
    candidates = feasible_nodes[candidates_mask]
    
    if len(candidates) == 1:
        return int(candidates[0])
        
    # Tie-breaker: Nearest to current
    dists_to_candidates = distance_matrix[current_node, candidates]
    best_candidate_idx = np.argmin(dists_to_candidates)
    
    return int(candidates[best_candidate_idx])
# EVOLVE-BLOCK-END
