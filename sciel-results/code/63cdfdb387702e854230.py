"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None

def _get_time_remaining():
    """Return seconds remaining before the 150s limit (with safety margin)."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elapsed = time.time() - _start_time
    return max(0.0, 145.0 - elapsed)


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Time guard: if we're running low on time, fall back to simple nearest neighbor
    if _get_time_remaining() < 1.0:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # If no unvisited nodes, this shouldn't happen but handle gracefully
    if len(unvisited_nodes) == 0:
        return destination_node
    
    # If only one unvisited node, return it
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # Score each candidate by: dist(current, candidate) + dist(candidate, destination)
    # This is a simple 2-opt lookahead heuristic
    d_from_current = distance_matrix[current_node, unvisited_nodes]
    d_to_destination = distance_matrix[unvisited_nodes, destination_node]
    scores = d_from_current + d_to_destination
    
    return int(unvisited_nodes[np.argmin(scores)])
# EVOLVE-BLOCK-END
