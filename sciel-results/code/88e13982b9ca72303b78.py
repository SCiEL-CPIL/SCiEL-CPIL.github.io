import numpy as np
import time

# Global time tracking for safety
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
        
    # If time is running out, fall back to simple nearest neighbor to ensure termination
    if _time_exceeded():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    # Lookahead Nearest Neighbor
    # For each candidate c in unvisited_nodes:
    # Cost = dist(current, c) + min(dist(c, v) for v in unvisited_nodes if v != c)
    # If c is the last node to be visited, the "next" step would be to destination.
    # However, since we don't know if c is the last, we assume there is always another unvisited node 
    # unless it's the last one.
    
    # Vectorized approach for speed:
    # 1. Get distances from current to all unvisited: d_curr
    # 2. Get distances from all unvisited to all unvisited: D_sub
    # 3. For each candidate i, we want min(D_sub[i, :]) excluding D_sub[i, i]
    
    d_curr = distance_matrix[current_node, unvisited_nodes]
    
    # Matrix of distances between unvisited nodes
    D_sub = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # We need the minimum distance from each unvisited node to any OTHER unvisited node.
    # Set diagonal to infinity, then take min along axis 1.
    np.fill_diagonal(D_sub, np.inf)
    min_next_dist = np.min(D_sub, axis=1)
    
    # Total lookahead cost
    scores = d_curr + min_next_dist
    
    # Pick the candidate with the minimum score
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
