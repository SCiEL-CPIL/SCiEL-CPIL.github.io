import numpy as np
import time

_start_time = time.time()
_LIMIT = 145.0  # Safety margin before 150s limit

def _is_deadline_passed():
    return (time.time() - _start_time) > _LIMIT

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Check deadline - fallback to pure nearest neighbor
    if _is_deadline_passed():
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
    
    # Compute distances from current to all candidates
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Compute distances from candidates to destination
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Adaptive alpha based on progress through the tour
    total_nodes = distance_matrix.shape[0]
    if total_nodes > 1:
        progress = 1.0 - (n_unvisited / total_nodes)
    else:
        progress = 1.0
    
    # Quadratic progression for stronger destination bias near the end
    alpha = 0.05 + 0.30 * (progress ** 2)
    
    # For small sets, use 2-step lookahead to improve quality
    if n_unvisited <= 15:
        # Distances between candidates
        D = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        np.fill_diagonal(D, np.inf)
        min_neighbor_dists = np.min(D, axis=1)
        
        # Combine: greedy + destination bias + 2-step lookahead
        # Normalize to make scale-invariant
        mean_dist = np.mean(dists_from_current)
        if mean_dist > 1e-10:
            scores = (dists_from_current / mean_dist + 
                      alpha * (dists_to_dest / mean_dist) + 
                      0.5 * (min_neighbor_dists / mean_dist))
        else:
            scores = dists_from_current + alpha * dists_to_dest + 0.5 * min_neighbor_dists
    else:
        # For larger sets, use lighter heuristic to save time
        # Normalize distances
        mean_dist = np.mean(dists_from_current)
        if mean_dist > 1e-10:
            scores = dists_from_current / mean_dist + alpha * (dists_to_dest / mean_dist)
        else:
            scores = dists_from_current + alpha * dists_to_dest
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
