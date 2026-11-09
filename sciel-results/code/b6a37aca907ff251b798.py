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
    
    Uses a hybrid strategy:
    - If only 1-2 unvisited nodes remain, go nearest (or the only option).
    - Otherwise, use a 1-step lookahead: for each candidate next node, estimate
      the cost of going to it plus the cheapest connection from there to any other
      unvisited node (or to destination if that's the last step).
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
    
    # For larger sets, use 1-step lookahead with destination awareness
    distances_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Compute for each candidate: dist(current->cand) + min over other unvisited (dist(cand->other))
    # This approximates the marginal cost of inserting this node
    lookahead_scores = np.empty(n_unvisited)
    
    for i in range(n_unvisited):
        node_i = unvisited_nodes[i]
        # Distance from current to candidate
        d_cur_to_i = distances_from_current[i]
        
        # Minimum distance from candidate to any other unvisited node
        if n_unvisited > 1:
            # Mask out self
            d_i_to_others = distance_matrix[node_i, unvisited_nodes].copy()
            d_i_to_others[i] = np.inf
            min_next = np.min(d_i_to_others)
        else:
            min_next = 0.0
        
        # Base score: current-to-candidate + expected next step
        score = d_cur_to_i + min_next
        
        # Gentle bias toward destination: if candidate is destination, slight preference
        # Actually, destination is typically the start node for closing the tour.
        # We want to avoid going to destination too early unless it's beneficial.
        # Instead, penalize going to destination early (unless it's one of last few)
        if node_i == destination_node and n_unvisited > 4:
            score *= 1.05  # slight penalty for going to destination too early
        
        lookahead_scores[i] = score
    
    # Pick the candidate with lowest lookahead score
    best_idx = int(np.argmin(lookahead_scores))
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
