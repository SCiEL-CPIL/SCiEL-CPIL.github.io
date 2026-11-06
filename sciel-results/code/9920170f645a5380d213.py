"""Traveling salesman constructive heuristic with 2-Step Lookahead and Regret."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state
_start_node = -1
_prev_len = 0
_start_time = None
_TIME_LIMIT = 115.0

def _reset_state():
    global _start_node, _prev_len, _start_time
    _start_node = -1
    _prev_len = 0
    _start_time = None

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _start_node, _prev_len, _start_time
    
    # Initialize on first call or new instance detection
    n_unvisited = len(unvisited_nodes)
    
    if _start_time is None:
        _start_time = time.time()
    
    # Detect new instance: if unvisited count increases, it's a new start
    if n_unvisited > _prev_len:
        _start_node = current_node
        _prev_len = n_unvisited
        
    # Time check
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        # Fallback to simple nearest neighbor
        if n_unvisited == 0:
            return -1
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
        
    if n_unvisited == 0:
        return -1
        
    if n_unvisited == 1:
        next_node = int(unvisited_nodes[0])
        _prev_len = n_unvisited - 1
        return next_node
        
    if n_unvisited == 2:
        a, b = unvisited_nodes[0], unvisited_nodes[1]
        start_node = _start_node
        # Option 1: a then b, then back to start
        cost1 = distance_matrix[current_node, a] + distance_matrix[a, b] + distance_matrix[b, start_node]
        # Option 2: b then a, then back to start
        cost2 = distance_matrix[current_node, b] + distance_matrix[b, a] + distance_matrix[a, start_node]
        if cost1 <= cost2:
            next_node = int(a)
        else:
            next_node = int(b)
        _prev_len = n_unvisited - 1
        return next_node

    # General case: n_unvisited >= 3
    # 1. Immediate costs
    imm_costs = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Sub-matrix of distances between unvisited nodes
    # We need min and second min for each row
    sub_matrix = distance_matrix[unvisited_nodes[:, np.newaxis], unvisited_nodes[np.newaxis, :]]
    np.fill_diagonal(sub_matrix, np.inf)
    
    n_cands = len(unvisited_nodes)
    
    # Find 1st and 2nd smallest distances for each node in unvisited set
    # Using argpartition for efficiency
    indices = np.argpartition(sub_matrix, 2, axis=1)[:, :2]
    rows = np.arange(n_cands)
    d_1 = sub_matrix[rows, indices[:, 0]]
    d_2 = sub_matrix[rows, indices[:, 1]]
    
    # Lookahead: min distance to any other unvisited node
    lookahead_costs = d_1
    
    # Regret: difference between 2nd and 1st nearest neighbor
    regrets = d_2 - d_1
    
    # Combine scores
    # We want to minimize: d(current, u) + d(u, nearest) - lambda * regret
    # High regret (large gap) -> subtract more -> lower score -> more likely to be picked.
    # This prioritizes nodes that have a "unique" close neighbor that we might lose.
    alpha = 1.0  # Weight for lookahead
    beta = 1.0   # Weight for regret
    
    scores = imm_costs + alpha * lookahead_costs - beta * regrets
    
    best_idx = np.argmin(scores)
    next_node = int(unvisited_nodes[best_idx])
    
    _prev_len = n_unvisited - 1
    return next_node

# EVOLVE-BLOCK-END
