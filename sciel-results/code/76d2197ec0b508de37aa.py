"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    if len(unvisited_nodes) == 0:
        return depot

    # Filter to feasible nodes (demand fits in remaining capacity)
    unvisited_arr = np.array(unvisited_nodes)
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    feasible_nodes = unvisited_arr[feasible_mask]

    if len(feasible_nodes) == 0:
        return depot

    # Approach: Pure Nearest Neighbor
    # Select the feasible node with the minimum distance from the current node.
    # This is a standard constructive heuristic for VRP.
    dist_from_current = distance_matrix[current_node, feasible_nodes]
    
    best_idx = np.argmin(dist_from_current)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
