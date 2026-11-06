"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic based on local regret/density.
    For each feasible node, calculate the 'gap' between its nearest and second-nearest
    neighbor among other feasible unvisited nodes.
    Score = Distance from current to node + Gap.
    Select the node with the minimum score.
    This prioritizes nodes that are close to the current position AND located in
    dense clusters (small gap), which helps avoid long detours in subsequent steps.
    """
    if len(unvisited_nodes) == 0:
        return depot

    # Get distances from current_node to all unvisited nodes
    dists_to_unvisited = distance_matrix[current_node, unvisited_nodes]
    
    # Filter to only feasible nodes (demand fits in remaining capacity)
    feasible_mask = demands[unvisited_nodes] <= rest_capacity
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        # No feasible customers left, go back to depot
        return depot

    # Get the feasible unvisited node indices (original node IDs)
    feasible_nodes = unvisited_nodes[feasible_indices]
    feasible_dists = dists_to_unvisited[feasible_indices]
    
    # If only one feasible node, go there
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])
        
    # If only two feasible nodes, local gap is not well-defined (no 2nd neighbor for either)
    # Just pick the nearest one
    if len(feasible_nodes) == 2:
        best_idx = np.argmin(feasible_dists)
        return int(feasible_nodes[best_idx])

    # Calculate local gap for each feasible node
    # Gap_i = Dist(i, 2nd nearest neighbor) - Dist(i, nearest neighbor)
    gaps = np.zeros(len(feasible_nodes))
    
    # Pre-compute distance matrix between feasible nodes for efficiency?
    # N is small (<=50), so O(N^2) is fine.
    feasible_node_ids = feasible_nodes
    
    for idx, node_i in enumerate(feasible_node_ids):
        # Get distances from node_i to all other feasible nodes
        # We need to exclude node_i itself
        dists_from_i = distance_matrix[node_i, feasible_node_ids]
        
        # Set distance to self to infinity to ignore it
        dists_from_i[idx] = np.inf
        
        # Find the two smallest distances
        if len(dists_from_i) < 2:
            # Should not happen if len(feasible_nodes) >= 3
            gaps[idx] = 0
        else:
            # Get the two smallest values
            sorted_dists = np.sort(dists_from_i)
            nearest_dist = sorted_dists[0]
            second_nearest_dist = sorted_dists[1]
            
            # The gap represents how "isolated" the nearest connection is.
            # A small gap means the node is in a dense cluster (good for avoiding detours).
            # A large gap means the node has a unique close neighbor but is otherwise far (riskier).
            gaps[idx] = second_nearest_dist - nearest_dist

    # Combine distance from current with the local gap
    # We want to minimize total cost.
    # Score = Dist_to_Current + Gap
    scores = feasible_dists + gaps
    
    # Pick the node with the minimum score
    best_idx = np.argmin(scores)
    
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
