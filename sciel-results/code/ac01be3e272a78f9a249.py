"""
TSP constructive heuristic: Improved Regret-2 with adaptive soft-min and destination awareness.

Strategy:
- Use regret-2 as the primary selection criterion (opportunity cost)
- Add a soft-min based global structure term (cheaper than MST)
- Add destination-awareness to avoid getting stuck far from the return point
- Adaptive weighting based on remaining nodes
- Fast fallbacks for small n and deadline enforcement
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TOTAL_LIMIT = 150.0
_MARGIN = 5.0

def _is_deadline_passed():
    return (time.time() - _start_time) > (_TOTAL_LIMIT - _MARGIN)

def _softmin(arr, temperature=1.0):
    """
    Compute a soft minimum of an array using log-sum-exp.
    Returns a value close to min(arr) but smooth.
    """
    if len(arr) == 0:
        return 0.0
    # Subtract min for numerical stability
    min_val = np.min(arr)
    if temperature <= 0:
        return min_val
    # Soft min = min - temperature * log(sum(exp(-(x - min)/temperature)))
    # This gives a value between min and the mean
    shifted = -(arr - min_val) / temperature
    # Clip to avoid overflow
    shifted = np.clip(shifted, -500, 0)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return min_val - temperature * log_sum

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour using a hybrid regret-2
    strategy with soft-min global structure and destination awareness.
    """
    n = len(unvisited_nodes)
    
    if n == 0:
        return -1
        
    if n == 1:
        return int(unvisited_nodes[0])
        
    if n == 2:
        n1, n2 = unvisited_nodes
        cost1 = distance_matrix[current_node, n1] + distance_matrix[n1, n2] + distance_matrix[n2, destination_node]
        cost2 = distance_matrix[current_node, n2] + distance_matrix[n2, n1] + distance_matrix[n1, destination_node]
        return int(n1 if cost1 <= cost2 else n2)
        
    # Deadline check
    if _is_deadline_passed():
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
    
    # Distance from current to all unvisited
    d_curr = distance_matrix[current_node, unvisited_nodes]
    
    # Distance matrix among unvisited
    d_sub = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # Distance from each unvisited to destination
    d_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # --- Regret-2 computation ---
    # For each candidate u, find the best 2nd choice
    # c1(u) = d(current, u) = d_curr[u]
    # c2(u) = min over v != u of (d(u, v) + d(v, destination))
    future_costs = d_sub + d_dest[None, :]
    np.fill_diagonal(future_costs, np.inf)
    c2 = np.min(future_costs, axis=1)
    regret = c2 - d_curr
    
    # --- Soft-min global structure term ---
    # For each candidate u, estimate the residual tour cost of the remaining nodes
    # using a soft-min over the distances from u to other remaining nodes
    # This is much cheaper than MST
    # residual(u) = softmin of d(u, v) for v in unvisited, v != u
    # We use a temperature that adapts to n
    temperature = max(0.5, 2.0 / np.sqrt(n))
    
    # For each u, compute softmin of distances to other unvisited nodes
    residual_costs = np.zeros(n)
    d_sub_copy = d_sub.copy()
    np.fill_diagonal(d_sub_copy, np.inf)
    
    for i in range(n):
        if n > 1:
            residual_costs[i] = _softmin(d_sub_copy[i], temperature)
        else:
            residual_costs[i] = 0.0
    
    # --- Destination awareness ---
    # Penalize nodes that are far from the destination to avoid ending up
    # in a position where the return trip is expensive
    dest_penalty = d_dest
    
    # --- Adaptive weights ---
    # When many nodes remain, rely more on regret and global structure
    # When few remain, rely more on immediate cost and destination
    if n > 20:
        alpha = 0.5   # regret weight
        beta = 0.3    # residual structure weight
        gamma = 0.2   # destination weight
    elif n > 10:
        alpha = 0.4
        beta = 0.3
        gamma = 0.3
    else:
        alpha = 0.3
        beta = 0.2
        gamma = 0.5
    
    # Normalize terms to be on similar scales
    # Use median as a robust scale estimate
    scale_regret = np.median(regret) + 1e-10
    scale_residual = np.median(residual_costs) + 1e-10
    scale_dest = np.median(dest_penalty) + 1e-10
    scale_curr = np.median(d_curr) + 1e-10
    
    regret_norm = regret / scale_regret
    residual_norm = residual_costs / scale_residual
    dest_norm = dest_penalty / scale_dest
    curr_norm = d_curr / scale_curr
    
    # Final composite score
    scores = curr_norm + alpha * regret_norm + beta * residual_norm + gamma * dest_norm
    
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
