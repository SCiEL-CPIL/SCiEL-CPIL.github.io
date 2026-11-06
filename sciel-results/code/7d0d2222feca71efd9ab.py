# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import permutations

# Global deadline guard
_START_TIME = None
_TIME_LIMIT = 115.0  # Safety margin from 120s

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

    # Check time before doing expensive work
    if _time_exceeded():
        # Fallback to nearest neighbor
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    # For small numbers of unvisited nodes, find the exact best next node
    # by evaluating all permutations of the remaining nodes.
    if n_unvisited <= 6:
        best_cost = float('inf')
        best_first_node = -1
        
        # Generate permutations
        for p in permutations(unvisited_nodes):
            # Path: current -> p[0] -> p[1] -> ... -> p[-1] -> destination
            cost = distance_matrix[current_node, p[0]]
            for i in range(len(p) - 1):
                cost += distance_matrix[p[i], p[i+1]]
            cost += distance_matrix[p[-1], destination_node]
            
            if cost < best_cost:
                best_cost = cost
                best_first_node = p[0]
                
        return int(best_first_node)

    # For larger sets, use a 2-step lookahead heuristic
    U = unvisited_nodes
    
    # Distance from current to all candidates
    d_curr = distance_matrix[current_node, U]
    
    # Distance matrix between all candidates
    # We want min distance from u to any other v in U
    sub = distance_matrix[np.ix_(U, U)].copy()
    np.fill_diagonal(sub, np.inf)
    min_next = np.min(sub, axis=1)
    
    # Score: cost to reach u + cost to reach next best node from u
    scores = d_curr + min_next
    
    best_idx = np.argmin(scores)
    return int(U[best_idx])

# EVOLVE-BLOCK-END
