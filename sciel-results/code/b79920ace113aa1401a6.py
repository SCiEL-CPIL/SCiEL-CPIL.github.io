"""Rewritten program: TSP constructive heuristic using destination-aware 2-step lookahead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track start time for deadline enforcement
_start_time = time.time()
_TOTAL_LIMIT = 150.0  # Evaluator limit
_MARGIN = 5.0         # Safety margin

def _is_deadline_passed():
    """Check if we are close to the time limit."""
    return (time.time() - _start_time) > (_TOTAL_LIMIT - _MARGIN)

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    
    Strategy: Destination-aware 2-step Lookahead.
    For each candidate node u in unvisited_nodes, we estimate the cost of
    visiting u next as:
    cost(u) = d(current, u) + min_{v in unvisited, v != u} [d(u, v) + beta * d(v, destination)]
    
    This incorporates the global constraint of returning to destination into
    the local decision, avoiding choices that lead to poor return paths.
    """
    n = len(unvisited_nodes)
    
    if n == 0:
        return -1
        
    if n == 1:
        return int(unvisited_nodes[0])
        
    # Enforce deadline: if time is running out, fall back to fast 1-step NN
    if _is_deadline_passed():
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
        
    # 1. Distance from current node to all unvisited candidates
    d_curr = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Distance matrix among unvisited nodes
    # d_sub[i, j] is distance between unvisited_nodes[i] and unvisited_nodes[j]
    d_sub = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # 3. Distance from each unvisited node to destination
    d_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # 4. For each candidate i, find the minimum over v != i of [d(u_i, v) + beta * d(v, dest)]
    # Set diagonal to infinity so a node doesn't pick itself as the "next next" node
    np.fill_diagonal(d_sub, np.inf)
    
    # Weight for destination distance in the lookahead
    beta = 0.5
    
    # Combine: for each pair (i, v), the cost is d_sub[i,v] + beta * d_dest[v]
    # Then take min over v for each i
    combined = d_sub + beta * d_dest[np.newaxis, :]
    min_next = np.min(combined, axis=1)
    
    # 5. Calculate total heuristic score
    scores = d_curr + min_next
    
    # 6. Select the candidate with the minimum score
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
