"""Combined TSP constructive heuristic: density-aware nearest-neighbor."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None
_TIME_LIMIT = 115.0

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
    
    if n_unvisited == 2:
        u1, u2 = int(unvisited_nodes[0]), int(unvisited_nodes[1])
        cost1 = distance_matrix[current_node, u1] + distance_matrix[u2, destination_node]
        cost2 = distance_matrix[current_node, u2] + distance_matrix[u1, destination_node]
        return u1 if cost1 <= cost2 else u2
    
    # Time guard: fall back to pure NN
    if _time_exceeded():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # Primary signal: distance from current node (nearest-neighbor)
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Density signal: for each candidate, find its nearest unvisited neighbor's distance.
    # A node with a very far nearest neighbor is "isolated" and visiting it now
    # may strand the rest of the cluster. Penalize isolated nodes lightly.
    # Compute pairwise distances among unvisited nodes (vectorized)
    sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    # Set diagonal to inf so argmin skips self
    np.fill_diagonal(sub_matrix, np.inf)
    nearest_unvisited_dist = np.min(sub_matrix, axis=1)
    
    # Scale the isolation penalty: normalize by the median nearest-neighbor distance
    # so the penalty is relative, not absolute
    median_nn = np.median(nearest_unvisited_dist)
    if median_nn > 1e-10:
        isolation_penalty = 0.15 * (nearest_unvisited_dist / median_nn)
    else:
        isolation_penalty = np.zeros(n_unvisited)
    
    # Composite score: primarily nearest-neighbor, with light isolation penalty
    scores = dist_from_current + isolation_penalty
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
