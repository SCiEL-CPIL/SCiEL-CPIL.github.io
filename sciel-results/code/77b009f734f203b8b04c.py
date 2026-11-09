"""Improved TSP constructive heuristic using vectorized 2-step lookahead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None
_DEADLINE_MARGIN = 5.0  # seconds to spare before the 150s limit

def _get_deadline():
    """Track elapsed time with a safety margin."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _START_TIME + 150.0 - _DEADLINE_MARGIN


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Pick the next node in a TSP tour construction.
    
    Uses a vectorized 2-step lookahead strategy:
    - If only 1-2 unvisited nodes remain, handle directly.
    - Otherwise, for each candidate next node, compute:
      dist(current->cand) + min over other unvisited (dist(cand->other))
    This is done in a vectorized manner for speed.
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Safety guard: if we're running low on time, fall back to fast nearest-neighbor
    if time.time() > _get_deadline():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        # Should not happen per contract, but handle gracefully
        return int(destination_node)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    if n_unvisited == 2:
        a, b = unvisited_nodes[0], unvisited_nodes[1]
        # Choose which to visit next based on total path cost
        cost_a = distance_matrix[current_node, a] + distance_matrix[a, b]
        cost_b = distance_matrix[current_node, b] + distance_matrix[b, a]
        if cost_a <= cost_b:
            return int(a)
        else:
            return int(b)
    
    # Vectorized 1-step lookahead
    # Get distances from current to all unvisited
    d_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Get distances from each unvisited to all other unvisited
    # This is a (n_unvisited, n_unvisited) matrix
    d_between = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # Set diagonal to inf so we don't pick self
    np.fill_diagonal(d_between, np.inf)
    
    # For each candidate i, the min distance from i to any other unvisited
    min_next = np.min(d_between, axis=1)
    
    # Total score for each candidate
    scores = d_from_current + min_next
    
    # Gentle bias: penalize going to destination too early
    dest_mask = (unvisited_nodes == destination_node)
    if n_unvisited > 4:
        scores[dest_mask] *= 1.05
    
    best_idx = int(np.argmin(scores))
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
