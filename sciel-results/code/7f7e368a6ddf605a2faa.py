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
    Uses a quadratic progression for stronger destination bias near the end.
    Also normalizes distances to make alpha scale-invariant.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Compute distances
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Estimate total nodes: current + unvisited + destination (if not already visited)
    # The destination is the starting node, so total = n_unvisited + 1 (for current) + possibly dest
    # More robust: use the size of the distance matrix as total nodes
    total_nodes = distance_matrix.shape[0]
    
    # Progress: 0 when all nodes unvisited, 1 when only 1 remains
    if total_nodes > 1:
        progress = 1.0 - (n_unvisited / total_nodes)
    else:
        progress = 1.0
    
    # Quadratic progression: stronger bias near the end
    # Alpha ranges from 0.05 to 0.35 with quadratic scaling
    alpha = 0.05 + 0.30 * (progress ** 2)
    
    # Normalize distances to make the heuristic scale-invariant
    # Use the mean distance as a normalizer
    mean_dist = np.mean(dists_from_current)
    if mean_dist > 1e-10:
        dists_from_current_norm = dists_from_current / mean_dist
        dists_to_dest_norm = dists_to_dest / mean_dist
    else:
        dists_from_current_norm = dists_from_current
        dists_to_dest_norm = dists_to_dest
    
    scores = dists_from_current_norm + alpha * dists_to_dest_norm
    
    # Check deadline with safety margin
    if time.time() > _get_deadline():
        # Fallback to nearest neighbor
        return int(unvisited_nodes[np.argmin(dists_from_current)])
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
