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

    # Approach: Nearest Neighbor Heuristic
    # Select the feasible node that minimizes the distance from the current node.
    # This is a greedy local search approach, distinct from savings-based methods.
    # It tends to produce short, compact routes by always moving to the closest
    # available customer.
    
    dist_current_to_nodes = distance_matrix[current_node, feasible_nodes]
    
    best_idx = np.argmin(dist_current_to_nodes)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
