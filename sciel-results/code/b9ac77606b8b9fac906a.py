"""Improved heuristic for Capacitated VRP using compact-route greedy selection."""
# EVOLVE-BLOCK-START
import numpy as np

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Handle empty unvisited list
    if len(unvisited_nodes) == 0:
        return depot

    # Convert to numpy array for vectorized operations
    unvisited = np.array(unvisited_nodes)
    
    # 1. Filter feasible nodes based on capacity
    # demands is aligned with node indices
    feasible_mask = np.array([demands[j] <= rest_capacity for j in unvisited])
    
    if not np.any(feasible_mask):
        return depot

    feasible_nodes = unvisited[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot

    # 2. Calculate scores
    # Objective: Minimize marginal cost of adding node j to the current route.
    # Approximate marginal cost = dist(current, j) + dist(j, depot) - dist(current, depot)
    # Since dist(current, depot) is constant for the current step, we minimize:
    # dist(current, j) + dist(j, depot)
    
    dist_from_current = distance_matrix[current_node, feasible_nodes]
    dist_from_depot = distance_matrix[depot, feasible_nodes]
    
    scores = dist_from_current + dist_from_depot
    
    # 3. Select the node with the minimum score
    # In case of ties (floating point precision), numpy argmin picks the first one.
    # To add robustness, we could add a tiny penalty based on demand or index, 
    # but usually the distance metric is distinct enough.
    
    best_idx = np.argmin(scores)
    next_node = feasible_nodes[best_idx]
    
    return int(next_node)
# EVOLVE-BLOCK-END
