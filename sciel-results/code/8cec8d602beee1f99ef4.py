"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for deadline guard (optional, likely not needed for 50 nodes)
_START_TIME = None
_DEADLINE = 115.0  # 5s margin before 120s

def _init_deadline():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_exceeded():
    global _START_TIME
    if _START_TIME is None:
        return False
    return (time.time() - _START_TIME) > _DEADLINE

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    _init_deadline()
    
    if len(unvisited_nodes) == 0:
        return depot

    # Convert to numpy array for fast indexing
    unvisited = np.array(unvisited_nodes, dtype=int)
    
    # 1. Filter feasible nodes based on capacity
    unvisited_demands = demands[unvisited]
    feasible_mask = unvisited_demands <= rest_capacity
    
    if not np.any(feasible_mask):
        return depot

    feasible_nodes = unvisited[feasible_mask]
    
    # If time is running out, fall back to simple nearest neighbor to ensure completion
    if _time_exceeded():
        dists = distance_matrix[current_node, feasible_nodes]
        return int(feasible_nodes[np.argmin(dists)])

    # 2. Calculate scores
    # Heuristic: Minimize d(current, next) + d(next, depot)
    # This encourages picking nodes that are close to the current position 
    # AND close to the depot, keeping routes compact.
    
    dist_current_to_next = distance_matrix[current_node, feasible_nodes]
    dist_next_to_depot = distance_matrix[feasible_nodes, depot]
    
    # We can weight these. A common choice is 1.0 for both.
    # Another option is to penalize large demands if capacity is low, 
    # but geometric proximity is usually the primary driver for distance reduction.
    
    scores = dist_current_to_next + dist_next_to_depot
    
    # Select the node with the minimum score
    best_idx = np.argmin(scores)
    
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
