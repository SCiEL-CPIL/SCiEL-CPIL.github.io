"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    if len(unvisited_nodes) == 0:
        return depot

    # Filter to feasible customers (demand fits in remaining capacity)
    feasible = []
    for node in unvisited_nodes:
        if demands[node] <= rest_capacity:
            feasible.append(node)

    if len(feasible) == 0:
        return depot

    feasible = np.array(feasible)
    
    # Score each feasible candidate:
    # score = d(current, j) + 0.5 * d(j, depot)
    # This balances the cost of going to j against the cost of returning to depot,
    # keeping routes compact and avoiding long tails.
    d_to_candidate = distance_matrix[current_node, feasible]
    d_candidate_to_depot = distance_matrix[feasible, depot]
    
    scores = d_to_candidate + 0.5 * d_candidate_to_depot
    
    best_idx = np.argmin(scores)
    return int(feasible[best_idx])
# EVOLVE-BLOCK-END
