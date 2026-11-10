"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_DEADLINE = time.time() + 145.0  # 5s margin from 150s limit

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Check time limit
    if time.time() > _DEADLINE:
        return depot

    if not unvisited_nodes:
        return depot

    # Filter nodes by capacity constraint
    feasible_mask = demands[unvisited_nodes] <= rest_capacity
    feasible_nodes = unvisited_nodes[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot

    # Calculate scores: d(current, j) + d(j, depot)
    # This heuristic approximates the savings by penalizing nodes that are far from the depot,
    # encouraging routes that are compact and don't require long returns.
    dist_to_current = distance_matrix[current_node, feasible_nodes]
    dist_to_depot = distance_matrix[depot, feasible_nodes]
    
    scores = dist_to_current + dist_to_depot
    
    # Find the index of the minimum score
    min_idx = np.argmin(scores)
    best_node = feasible_nodes[min_idx]
    
    return int(best_node)
# EVOLVE-BLOCK-END
