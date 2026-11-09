"""Savings-based heuristic for Capacitated Vehicle Routing Problem."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 110.0  # Safety margin before 120s limit
_DEADLINE = _start_time + _TIMEOUT

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node using a greedy savings heuristic.
    Returns 0 (depot) if no feasible node can be added, if savings are negative,
    or if the time limit is reached.
    """
    if time.time() > _DEADLINE:
        return depot

    if len(unvisited_nodes) == 0:
        return depot

    # Convert to numpy array for vectorized operations
    unvisited = np.array(unvisited_nodes, dtype=int)
    
    # 1. Filter by capacity
    # demands is expected to be indexable by node id
    feasible_mask = demands[unvisited] <= rest_capacity
    feasible_nodes = unvisited[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot

    # 2. If at depot, pick the nearest feasible node
    if current_node == depot:
        dists = distance_matrix[current_node, feasible_nodes]
        min_idx = np.argmin(dists)
        return int(feasible_nodes[min_idx])

    # 3. Calculate savings for each feasible node
    # Saving S_j = d(current, depot) + d(depot, j) - d(current, j)
    d_current_depot = distance_matrix[current_node, depot]
    d_depot_j = distance_matrix[depot, feasible_nodes]
    d_current_j = distance_matrix[current_node, feasible_nodes]
    
    savings = d_current_depot + d_depot_j - d_current_j
    
    # 4. If the maximum saving is negative, it is cheaper to close the route
    if np.max(savings) < 0:
        return depot
    
    # 5. Pick the node with the maximum saving
    # If there are ties, argmax picks the first one. 
    # To break ties by distance (prefer closer if savings are identical), 
    # we can use a composite score, but simple argmax is a standard greedy step.
    max_saving_idx = np.argmax(savings)
    
    return int(feasible_nodes[max_saving_idx])
# EVOLVE-BLOCK-END
