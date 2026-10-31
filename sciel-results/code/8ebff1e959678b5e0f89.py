"""Rewritten program: improved TSP constructive heuristic with adaptive lookahead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()

def _get_deadline():
    """Return a deadline that is 3 seconds before the 150s limit (safety margin)."""
    return _start_time + 145.0

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    
    Strategy: Adaptive 1-step lookahead with dynamic alpha based on remaining nodes.
    As fewer nodes remain, increase the weight on distance-to-destination to ensure
    the tour closes efficiently.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Compute distances
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Adaptive alpha: start small, increase as we approach the end
    # When many nodes remain, alpha is small (mostly nearest neighbor)
    # When few nodes remain, alpha is larger (stronger destination bias)
    total_nodes = 50  # Known from task description
    progress = 1.0 - (n_unvisited / total_nodes)
    # Alpha ranges from 0.01 to 0.15
    alpha = 0.01 + 0.14 * progress
    
    scores = dists_from_current + alpha * dists_to_dest
    
    # Check deadline with safety margin
    if time.time() > _get_deadline():
        # Fallback to nearest neighbor
        return int(unvisited_nodes[np.argmin(dists_from_current)])
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
