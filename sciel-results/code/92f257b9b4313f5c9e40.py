"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    if not unvisited_nodes:
        return depot

    # Ensure unvisited_nodes is a numpy array for vectorized operations
    unvisited_arr = np.array(unvisited_nodes)
    
    # Filter for feasible nodes (demand <= rest_capacity)
    # This is a safety check. If the evaluator pre-filters, this is a no-op.
    # If not, this prevents capacity violations.
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    feasible_nodes = unvisited_arr[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot

    # Calculate distances
    # d(i, j) for all feasible j
    dist_from_current = distance_matrix[current_node, feasible_nodes]
    # d(j, depot) for all feasible j
    dist_to_depot = distance_matrix[feasible_nodes, depot]
    
    # Heuristic Score:
    # We want to minimize the immediate travel distance, but also keep the route
    # compact around the depot to reduce the return trip cost.
    # Score = dist(i, j) + lambda * dist(j, depot)
    # lambda = 0.5 balances immediate cost with future return cost.
    lambda_val = 0.5
    scores = dist_from_current + lambda_val * dist_to_depot
    
    # Select the node with the minimum score
    min_idx = np.argmin(scores)
    next_node = feasible_nodes[min_idx]
    
    return int(next_node)
# EVOLVE-BLOCK-END
