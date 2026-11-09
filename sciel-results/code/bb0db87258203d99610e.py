"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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
    
    Uses a vectorized 2-step lookahead with destination awareness:
    - For each candidate, estimate the cost of going to it plus the cheapest
      connection from there to any other unvisited node.
    - This is computed in a single matrix operation for speed.
    - Also incorporate a gentle bias toward the destination node to help close
      the tour.
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
    
    # For larger sets, use vectorized 2-step lookahead
    # distances_from_current: (n_unvisited,)
    distances_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # d_i_to_others: (n_unvisited, n_unvisited) - distance from each candidate to all others
    d_i_to_others = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # Mask out diagonal (self)
    np.fill_diagonal(d_i_to_others, np.inf)
    
    # min_next: (n_unvisited,) - minimum distance from each candidate to any other
    min_next = np.min(d_i_to_others, axis=1)
    
    # Base score: current-to-candidate + expected next step
    scores = distances_from_current + min_next
    
    # Gentle bias toward destination: penalize going to destination early
    # Find index of destination_node in unvisited_nodes
    dest_mask = (unvisited_nodes == destination_node)
    if np.any(dest_mask) and n_unvisited > 4:
        scores[dest_mask] *= 1.05  # slight penalty for going to destination too early
    
    # Pick the candidate with lowest score
    best_idx = int(np.argmin(scores))
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
