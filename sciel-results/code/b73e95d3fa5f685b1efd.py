import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Check deadline
    if time.time() - _START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return depot

    if len(unvisited_nodes) == 0:
        return depot

    # Ensure numpy arrays for efficient indexing
    unvisited_arr = np.asarray(unvisited_nodes)
    
    # Get demands for these nodes
    # demands is expected to be indexable by node id
    try:
        node_demands = demands[unvisited_arr]
    except Exception:
        # Fallback if demands is a list and unvisited_arr is array of ints
        node_demands = np.array([demands[i] for i in unvisited_arr])

    # Feasibility: demand <= rest_capacity
    feasible_mask = node_demands <= rest_capacity
    
    if not np.any(feasible_mask):
        return depot

    feasible_nodes = unvisited_arr[feasible_mask]
    
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])

    # Calculate Savings Score
    # S(v) = d(current, depot) + d(depot, v) - d(current, v)
    # Maximize S(v)
    
    # Ensure distance_matrix is numpy array for fancy indexing
    if not isinstance(distance_matrix, np.ndarray):
        distance_matrix = np.asarray(distance_matrix)

    d_current_depot = distance_matrix[current_node, depot]
    d_depot_feasible = distance_matrix[depot, feasible_nodes]
    d_current_feasible = distance_matrix[current_node, feasible_nodes]
    
    savings = d_current_depot + d_depot_feasible - d_current_feasible
    
    # Find max savings
    max_savings = np.max(savings)
    
    # Get all nodes with max savings
    # Using a small epsilon for floating point comparison if necessary, 
    # but exact equality on float arrays from same ops is usually stable for ties.
    best_indices = np.where(savings == max_savings)[0]
    
    if len(best_indices) == 1:
        return int(feasible_nodes[best_indices[0]])
    
    # Tie-breaker: Choose the one with minimum distance from current
    best_nodes = feasible_nodes[best_indices]
    d_current_best = distance_matrix[current_node, best_nodes]
    min_dist_idx = np.argmin(d_current_best)
    
    return int(best_nodes[min_dist_idx])
