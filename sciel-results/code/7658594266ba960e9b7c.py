"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state for time management and caching
_start_time = None
_time_limit = 115.0  # Safety margin from 120s limit
_cache = {}

def _ensure_start_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()

def _time_exceeded():
    global _start_time
    if _start_time is None:
        return False
    return (time.time() - _start_time) > _time_limit

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    _ensure_start_time()
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return int(current_node)
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    if n_unvisited == 2:
        # Choose the one that minimizes total edge cost to close the tour
        # Option 1: current -> u1 -> u2 -> dest
        # Option 2: current -> u2 -> u1 -> dest
        u1, u2 = unvisited_nodes
        cost1 = distance_matrix[current_node, u1] + distance_matrix[u1, u2] + distance_matrix[u2, destination_node]
        cost2 = distance_matrix[current_node, u2] + distance_matrix[u2, u1] + distance_matrix[u1, destination_node]
        if cost1 <= cost2:
            return int(u1)
        else:
            return int(u2)

    # Check time
    if _time_exceeded():
        # Fallback to nearest neighbor
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    # Get distances from current node to all unvisited
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Get distances from unvisited to destination
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Compute a composite score
    # We want to minimize: dist(current, u) + dist(u, dest)
    # But also consider the "angle" or geometry to avoid crossing?
    # A simple effective score is the sum of distances.
    # Let's try a weighted sum: w1 * dist(current, u) + w2 * dist(u, dest)
    # Usually w1 = w2 = 0.5 is a good start (minimizing the 2-opt move cost locally).
    # Actually, minimizing dist(current, u) + dist(u, dest) is exactly the 2-opt lookahead.
    
    scores = dist_from_current + dist_to_dest
    
    # Find the best candidate
    best_idx = np.argmin(scores)
    best_node = unvisited_nodes[best_idx]
    
    return int(best_node)

# EVOLVE-BLOCK-END
