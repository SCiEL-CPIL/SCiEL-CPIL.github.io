import numpy as np
import time

_START_TIME = None
_TIME_LIMIT = 110.0

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
    
    if _time_exceeded():
        # Fallback to NN
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
        
    # Get distances from current to all unvisited
    d_current = distance_matrix[current_node, unvisited_nodes]
    
    # Get distances from destination to all unvisited
    d_dest = distance_matrix[destination_node, unvisited_nodes]
    
    # Get sub-matrix of distances between unvisited nodes
    d_unvisited = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    scores = np.zeros(n_unvisited)
    
    # For each candidate c, calculate:
    # score = d_current[c] + min( d_unvisited[c, :] + d_dest[:] ) 
    # Note: The min is over n != c. 
    # Since d_unvisited[c, c] is 0, and d_dest[c] is positive, 
    # if n_unvisited > 1, the min will likely not be at c unless d_dest is 0 (impossible) or other distances are huge.
    # However, strictly speaking, we should exclude c from the minimization.
    # But d_unvisited[c, c] = 0. So term is 0 + d_dest[c].
    # Is it possible that going c -> c -> dest is better than c -> n -> dest?
    # No, because we must visit all nodes. If we pick c, we can't pick c again.
    # So the next node n must be in U \ {c}.
    
    for i in range(n_unvisited):
        # Exclude i from the options for n
        # We want min over j != i of (d_unvisited[i, j] + d_dest[j])
        
        # Vectorized approach:
        # Create a mask or just handle the diagonal.
        # Since d_unvisited[i, i] = 0, the term is d_dest[i].
        # If d_dest[i] is smaller than any d_unvisited[i, j] + d_dest[j], 
        # it would incorrectly suggest staying at i (which is invalid).
        # However, d_unvisited[i, j] >= 0.
        # Usually d_dest[i] is not the minimum if there are other close nodes.
        # To be safe and correct:
        
        costs = d_unvisited[i, :] + d_dest[:]
        # Set cost[i] to infinity to exclude self
        costs[i] = np.inf
        min_next_cost = np.min(costs)
        
        scores[i] = d_current[i] + min_next_cost
        
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
