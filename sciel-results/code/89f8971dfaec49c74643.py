"""Combined TSP constructive heuristic: adaptive destination-bias with smooth surrogate scoring."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()

def _get_deadline():
    """Return a deadline that is 2 seconds before the 120s limit (safety margin)."""
    return _start_time + 118.0

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    
    Strategy: Adaptive weighted combination of nearest-neighbor and distance-to-destination.
    - Early in the tour: mostly nearest-neighbor (alpha small)
    - Late in the tour: stronger destination bias to close the tour efficiently
    - Uses normalization for scale invariance
    - Adaptive alpha with smooth progression
    
    The score for each candidate is:
        d(current, cand) / mean_d_current + alpha * d(cand, destination) / mean_d_dest
    
    where alpha adapts based on progress through the tour.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Compute distances from current node to all unvisited nodes
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Compute distances from unvisited nodes to destination
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Estimate total nodes from distance matrix
    total_nodes = distance_matrix.shape[0]
    
    # Progress: 0 when all nodes unvisited, 1 when only 1 remains
    if total_nodes > 1:
        progress = 1.0 - (n_unvisited / total_nodes)
    else:
        progress = 1.0
    
    # Check deadline early - if we're running out of time, fall back to nearest neighbor
    if time.time() > _get_deadline():
        return int(unvisited_nodes[np.argmin(dists_from_current)])
    
    # Adaptive alpha: use a smooth curve that increases with progress
    # Quadratic gives stronger bias near the end of the tour
    alpha = 0.05 + 0.40 * (progress ** 2)
    
    # Normalize distances by their mean to make alpha scale-invariant
    mean_from = np.mean(dists_from_current)
    mean_to = np.mean(dists_to_dest)
    
    if mean_from > 1e-10 and mean_to > 1e-10:
        dists_from_norm = dists_from_current / mean_from
        dists_to_norm = dists_to_dest / mean_to
    else:
        dists_from_norm = dists_from_current
        dists_to_norm = dists_to_dest
    
    # Weighted score: combine nearest-neighbor and destination-bias
    scores = dists_from_norm + alpha * dists_to_norm
    
    # Final deadline check
    if time.time() > _get_deadline():
        return int(unvisited_nodes[np.argmin(dists_from_current)])
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
