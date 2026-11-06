"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 3.0


def _time_exceeded():
    return (time.time() - _START_TIME) > (_DEADLINE - _SAFETY_MARGIN)


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic for capacitated VRP: pick next customer or depot (0) to close route.
    
    Strategy:
      1. Filter unvisited nodes that fit in rest_capacity.
      2. If none fit, return depot.
      3. Score each feasible node using a lookahead metric:
         - immediate distance from current_node
         - plus a fraction of the minimum distance from that node to any other feasible node or depot
         This avoids dead-end picks where a node is close now but far from everything else.
    """
    if _time_exceeded():
        # Safety: return depot to close route quickly
        return 0

    unvisited = list(unvisited_nodes)
    if not unvisited:
        return 0

    # Filter by capacity
    feasible = [n for n in unvisited if demands[n] <= rest_capacity]
    if not feasible:
        return 0

    # Convert to numpy arrays for vectorized operations
    feasible_arr = np.array(feasible)
    dist_from_current = distance_matrix[current_node, feasible_arr]

    # Lookahead: for each feasible node, compute min distance to any other feasible node or depot
    # We want to avoid nodes that are isolated (far from everything else)
    # Compute distances from each feasible node to all feasible nodes + depot
    # To keep it fast, we do a simple approach: for each feasible node, find min distance to depot or other feasible
    
    n_feasible = len(feasible_arr)
    if n_feasible == 1:
        return int(feasible_arr[0])

    # Build a small distance matrix among feasible nodes + depot
    # indices: feasible_arr plus depot
    indices = np.concatenate([feasible_arr, [depot]])
    sub_dist = distance_matrix[np.ix_(indices, indices)]
    
    # For each feasible node, compute:
    # 1. min distance to any other feasible node (continuation cost)
    # 2. distance to depot (closing cost)
    # We weight these to balance route continuation vs. closing efficiency
    
    lookahead_min = np.zeros(n_feasible)
    dist_to_depot = np.zeros(n_feasible)
    for i in range(n_feasible):
        # min distance from feasible node i to any other feasible node (exclude self)
        # Only consider other feasible nodes (indices 0..n_feasible-1 excluding i)
        other_feasible = [j for j in range(n_feasible) if j != i]
        if other_feasible:
            lookahead_min[i] = np.min(sub_dist[i, other_feasible])
        else:
            lookahead_min[i] = sub_dist[i, n_feasible]  # only depot available
        
        # Distance to depot
        dist_to_depot[i] = sub_dist[i, n_feasible]

    # Capacity pressure: if picking this node leaves very little capacity, 
    # we might be forced to close soon, so penalize slightly
    remaining_after = rest_capacity - demands[feasible_arr]
    # Normalize: if remaining is small relative to rest_capacity, add penalty
    capacity_penalty = np.where(remaining_after < 5, (5 - remaining_after) * 0.1, 0)

    # Score: immediate distance + alpha * min_continuation + beta * dist_to_depot + capacity_penalty
    # alpha controls how much we care about not being in a dead-end
    # beta controls how much we care about being able to close efficiently
    alpha = 0.3
    beta = 0.15
    scores = dist_from_current + alpha * lookahead_min + beta * dist_to_depot + capacity_penalty

    # Pick the node with minimum score
    best_idx = np.argmin(scores)
    return int(feasible_arr[best_idx])
# EVOLVE-BLOCK-END
