"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for deadline guard
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
    unvisited = np.array(unvisited_nodes, dtype=np.int64)
    
    # Filter by capacity
    # demands is likely a list or array indexed by node id
    # We need to check demands[node] <= rest_capacity
    feasible_mask = np.array([demands[node] <= rest_capacity for node in unvisited], dtype=bool)
    
    if not np.any(feasible_mask):
        return depot
        
    feasible_nodes = unvisited[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot

    # Calculate scores: d(i, j) - d(depot, j)
    # This is the "Savings" heuristic adapted for sequential insertion
    # Minimizing this maximizes the savings of appending j to the current route
    # compared to starting a new route for j.
    
    # d(i, j)
    dists_from_current = distance_matrix[current_node, feasible_nodes]
    # d(depot, j)
    dists_from_depot = distance_matrix[depot, feasible_nodes]
    
    scores = dists_from_current - dists_from_depot
    
    # Pick the one with minimum score
    min_idx = np.argmin(scores)
    next_node = feasible_nodes[min_idx]
    
    return int(next_node)
# EVOLVE-BLOCK-END
