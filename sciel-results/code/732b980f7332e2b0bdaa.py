"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = None
_TIMEOUT = 115.0
_DEADLINE = None


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic to pick the next customer or return to depot.
    
    Uses a savings-based scoring function that prioritizes nodes far from depot
    to create longer routes and reduce vehicle count, while still respecting
    capacity constraints.
    """
    global _start_time, _DEADLINE
    
    if _start_time is None:
        _start_time = time.time()
        _DEADLINE = _start_time + _TIMEOUT
    
    # Deadline guard
    if time.time() > _DEADLINE:
        return int(depot)
    
    # Convert unvisited_nodes to numpy array if it's a list
    if isinstance(unvisited_nodes, (list, tuple)):
        unvisited_arr = np.array(unvisited_nodes)
    else:
        unvisited_arr = unvisited_nodes
    
    # If no unvisited nodes, return depot
    if len(unvisited_arr) == 0:
        return int(depot)
    
    # Find feasible nodes: those whose demand fits in rest_capacity
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    
    # If no feasible nodes, return depot
    if not np.any(feasible_mask):
        return int(depot)
    
    feasible_nodes = unvisited_arr[feasible_mask]
    
    # Compute distances
    dists_from_current = distance_matrix[current_node, feasible_nodes]
    dists_to_depot = distance_matrix[feasible_nodes, depot]
    dist_current_to_depot = distance_matrix[current_node, depot]
    
    # Enhanced savings-based score with capacity-awareness
    # saving(j) = dist(current, depot) + dist(depot, j) - dist(current, j)
    # We want to MAXIMIZE savings, so we minimize negative savings.
    #
    # score(j) = dist(current, j) - alpha * saving(j) + beta * capacity_penalty
    # where capacity_penalty encourages using remaining capacity efficiently
    
    alpha = 0.7
    beta = 0.1
    
    savings = dist_current_to_depot + dists_to_depot - dists_from_current
    base_scores = dists_from_current - alpha * savings
    
    # Capacity penalty: prefer nodes that use a reasonable fraction of remaining capacity
    # This helps pack routes more efficiently
    demand_ratio = demands[feasible_nodes] / rest_capacity
    # Penalize very small demands (waste capacity) and very large demands (might strand us)
    # Ideal is around 0.5-0.7 of remaining capacity
    capacity_penalty = beta * (demand_ratio - 0.6) ** 2
    
    scores = base_scores + capacity_penalty
    
    # Pick the node with the minimum score
    best_idx = np.argmin(scores)
    best_node = feasible_nodes[best_idx]
    
    return int(best_node)
# EVOLVE-BLOCK-END
