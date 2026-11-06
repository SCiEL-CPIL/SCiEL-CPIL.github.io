"""Traveling salesman constructive heuristic with 2-step lookahead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track the tour being built
_current_tour = []
_start_time = None
_TIME_LIMIT = 110.0  # Safety margin for 120s/150s limit

def _reset_state():
    global _current_tour, _start_time
    _current_tour = []
    _start_time = time.time()

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _current_tour, _start_time
    
    # Initialize on first call
    if _start_time is None:
        _start_time = time.time()
        _current_tour = [current_node]
    
    # Time check
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        # Fallback to simple nearest neighbor to finish quickly
        if len(unvisited_nodes) == 0:
            return -1
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    if len(unvisited_nodes) == 0:
        return -1
    
    # If only one node left, must pick it.
    if len(unvisited_nodes) == 1:
        next_node = int(unvisited_nodes[0])
        _current_tour.append(next_node)
        return next_node
    
    # If two nodes left, pick the one that gives shorter closing
    if len(unvisited_nodes) == 2:
        a, b = unvisited_nodes[0], unvisited_nodes[1]
        start_node = _current_tour[0]
        # Option 1: a then b, then back to start
        cost1 = distance_matrix[current_node, a] + distance_matrix[a, b] + distance_matrix[b, start_node]
        # Option 2: b then a, then back to start
        cost2 = distance_matrix[current_node, b] + distance_matrix[b, a] + distance_matrix[a, start_node]
        if cost1 <= cost2:
            next_node = int(a)
        else:
            next_node = int(b)
        _current_tour.append(next_node)
        return next_node
    
    n_cands = len(unvisited_nodes)
    
    # 1. Immediate costs from current_node to all candidates
    imm_costs = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Compute the cost of the *next* step for each candidate.
    # For each candidate u, we want to know min_{v in unvisited, v != u} d(u, v).
    # We can compute the sub-matrix of distances between unvisited nodes.
    sub_matrix = distance_matrix[unvisited_nodes[:, np.newaxis], unvisited_nodes[np.newaxis, :]]
    
    # Mask the diagonal (distance to self) with infinity
    np.fill_diagonal(sub_matrix, np.inf)
    
    # For each row (candidate u), find the minimum distance to any other unvisited node.
    # This represents the "lookahead" cost.
    lookahead_costs = np.min(sub_matrix, axis=1)
    
    # 3. Combine Immediate Cost and Lookahead Cost.
    # Score = d(current, u) + lambda * d(u, best_next)
    # We want to minimize this.
    # Lambda = 0.5 is a reasonable weight to balance immediate proximity with future flexibility.
    # If Lambda is too high, we might pick a node that is close now but leads to a very far next step.
    # If Lambda is too low, it's just nearest neighbor.
    # Let's try Lambda = 0.5.
    lambda_weight = 0.5
    scores = imm_costs + lambda_weight * lookahead_costs
    
    # Pick the candidate with minimum score
    best_idx = np.argmin(scores)
    best_candidate = int(unvisited_nodes[best_idx])
    
    # Update global state
    _current_tour.append(best_candidate)
    
    return best_candidate

# EVOLVE-BLOCK-END
