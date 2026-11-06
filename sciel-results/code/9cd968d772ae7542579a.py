"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None
_TIME_LIMIT = 115.0  # Safety margin from 120s limit


def _ensure_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()


def _time_exceeded():
    global _START_TIME
    if _START_TIME is None:
        return False
    return (time.time() - _START_TIME) > _TIME_LIMIT


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    _ensure_start_time()
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return int(current_node)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # If time is running out, fall back to nearest neighbor for safety
    if _time_exceeded():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # For small sets, use nearest neighbor (fast and good enough)
    if n_unvisited <= 3:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # Compute distances from current to all unvisited
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Compute distances from each unvisited to destination
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Compute pairwise distances between unvisited nodes
    # This is O(n^2) where n = n_unvisited
    # For 50 cities, this is manageable
    
    # Use a combined score: 
    # Primary: distance from current to candidate
    # Secondary: distance from candidate to destination (to avoid bad endings)
    # Tertiary: how "central" the candidate is among remaining nodes (to keep options open)
    
    # Simple but effective heuristic: 
    # score = dist_from_current + alpha * dist_to_dest
    # where alpha balances immediate cost vs. future cost
    
    # Adaptive alpha: when few nodes remain, weight destination more heavily
    if n_unvisited <= 5:
        alpha = 0.5
    elif n_unvisited <= 10:
        alpha = 0.3
    else:
        alpha = 0.1
    
    scores = dist_from_current + alpha * dist_to_dest
    
    # Find the minimum score
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
