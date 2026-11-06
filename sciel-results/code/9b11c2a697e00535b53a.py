"""Traveling salesman constructive heuristic with Regret-2 + 2-step lookahead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track the tour being built
_current_tour = []
_start_time = None
_TIME_LIMIT = 115.0  # Safety margin for 120s limit

def _reset_state():
    global _current_tour, _start_time
    _current_tour = []
    _start_time = None

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
    start_node = _current_tour[0]
    
    # 1. Immediate costs from current_node to all candidates
    imm_costs = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Compute the sub-matrix of distances between unvisited nodes
    sub_matrix = distance_matrix[unvisited_nodes[:, np.newaxis], unvisited_nodes[np.newaxis, :]]
    
    # Mask the diagonal (distance to self) with infinity
    np.fill_diagonal(sub_matrix, np.inf)
    
    # --- Regret-2 Component ---
    # Find the smallest and second smallest distance for each node
    indices = np.argpartition(sub_matrix, 2, axis=1)[:, :2]
    rows = np.arange(n_cands)
    d_1 = sub_matrix[rows, indices[:, 0]]
    d_2 = sub_matrix[rows, indices[:, 1]]
    regrets = d_2 - d_1
    
    # --- 2-Step Lookahead Component ---
    # For each candidate u, estimate the cost of the next step.
    # We approximate the future cost by looking at the minimum distance from u to any other unvisited node v.
    # This is a proxy for how "isolated" or "connected" u is to the rest of the unvisited set.
    # A lower min_dist_to_other suggests u is well-connected, potentially reducing future costs.
    # However, a more direct 2-step lookahead would be:
    # Cost(u) = d(current, u) + min_{v != u} d(u, v)
    # This is what we are effectively doing by using d_1 (which is min_{v != u} d(u, v)).
    
    # Let's refine the score.
    # Standard Regret-2 minimizes: d(current, u) - alpha * regret(u)
    # But we can also incorporate the "next best" connection cost.
    
    # Let's try a hybrid score:
    # Score = imm_costs + beta * d_1 - alpha * regrets
    # d_1 is the cost to the nearest neighbor. Minimizing this encourages picking nodes that are close to the cluster.
    # Regrets penalizes nodes that are "urgent" (high regret means we lose a lot if we don't pick them now).
    
    # Let's tune alpha and beta.
    # In the previous version, alpha=1.0 was used.
    # Let's try alpha=1.0 and beta=0.5.
    
    alpha = 1.0
    beta = 0.5
    
    scores = imm_costs + beta * d_1 - alpha * regrets
    
    # Pick the candidate with minimum score
    best_idx = np.argmin(scores)
    best_candidate = int(unvisited_nodes[best_idx])
    
    # Update global state
    _current_tour.append(best_candidate)
    
    return best_candidate

# EVOLVE-BLOCK-END
