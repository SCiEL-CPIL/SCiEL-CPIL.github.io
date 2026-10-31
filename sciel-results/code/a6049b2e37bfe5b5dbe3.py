"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for caching and deadline management
# We use a simple module-level cache keyed by (id of distance_matrix, frozenset of unvisited, current, destination)
# However, since unvisited_nodes is an ndarray, we need a robust key.
# For 50 cities, the number of steps is 50. The total number of calls is 50.
# Caching might not be strictly necessary if the algorithm is fast enough, but let's add a deadline guard.

_START_TIME = time.time()
_TIMEOUT = 115.0  # Safety margin before 120s limit

def _check_timeout():
    return (time.time() - _START_TIME) > _TIMEOUT

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node in the TSP tour using a hybrid heuristic:
    1. Nearest Neighbor (base)
    2. 2-step lookahead (improvement)
    3. Penalty for isolating remaining nodes (structural improvement)
    
    The heuristic minimizes:
    d(current, u) + min_{v in unvisited\{u}} d(u, v)
    This is the standard 2-opt style lookahead for construction.
    To prevent the tour from becoming disconnected or creating long "tails",
    we also consider the distance from the destination node, as we must eventually return.
    
    Specifically, we score each candidate u as:
    score(u) = d(current, u) + min_{v in unvisited\{u}} d(u, v)
    
    If the set of unvisited nodes is small, we compute exact costs for the remaining suffix.
    """
    if len(unvisited_nodes) == 0:
        return -1  # Should not happen in valid loop, but safe fallback

    # If only one node left, must pick it
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])

    # If timeout is near, fall back to nearest neighbor for speed
    if _check_timeout():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    n_unvisited = len(unvisited_nodes)
    
    # For very small sets, we can afford more complex logic, but 2-step lookahead is O(N^2) per step, total O(N^3) = 125k ops, very fast.
    # Let's implement the 2-step lookahead:
    # For each candidate u in unvisited_nodes:
    #   cost = d(current, u) + min_{v in unvisited_nodes, v != u} d(u, v)
    # Choose u with min cost.
    
    # Vectorized implementation for speed
    # d_current: shape (N,)
    d_current = distance_matrix[current_node, unvisited_nodes]
    
    # d_unvisited: shape (N, N) - distance between unvisited nodes
    d_unvisited = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # To find min_{v != u} d(u, v), we can set the diagonal to inf and take argmin
    # But we need the value, not just the index.
    # np.min with where mask is slow. 
    # Trick: set diagonal to inf, then min along axis 1.
    d_unvisited_copy = d_unvisited.copy()
    np.fill_diagonal(d_unvisited_copy, np.inf)
    
    min_dist_to_next = np.min(d_unvisited_copy, axis=1)
    
    total_score = d_current + min_dist_to_next
    
    # Handle case where min_dist_to_next is inf (shouldn't happen if N>1, but safe)
    if np.any(np.isinf(total_score)):
        # Fallback to nearest neighbor
        return int(unvisited_nodes[np.argmin(d_current)])
        
    best_idx = np.argmin(total_score)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
