"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for beam search
_beam_state = {
    'tours': [],  # List of (tour_so_far, current_node, unvisited_set, cost)
    'beam_width': 4,
    'start_time': None,
    'deadline': 115.0  # Safety margin before 120s limit
}

def _init_beam_state():
    """Initialize beam search state."""
    _beam_state['tours'] = []
    _beam_state['start_time'] = time.time()

def _is_time_up():
    """Check if we're approaching the deadline."""
    if _beam_state['start_time'] is None:
        return False
    return (time.time() - _beam_state['start_time']) > _beam_state['deadline']

def _estimate_remaining_cost(current_node, destination_node, unvisited_nodes, distance_matrix):
    """Estimate remaining tour cost using minimum spanning tree approximation."""
    if len(unvisited_nodes) == 0:
        return distance_matrix[current_node, destination_node]
    
    # Simple estimate: sum of nearest neighbor distances for unvisited + return to destination
    # Use a cheaper heuristic: average nearest neighbor distance * count
    if len(unvisited_nodes) == 1:
        return distance_matrix[current_node, unvisited_nodes[0]] + distance_matrix[unvisited_nodes[0], destination_node]
    
    # For each unvisited node, find its nearest neighbor in unvisited set + destination
    # This is O(n^2) which is acceptable for n<=50
    remaining = unvisited_nodes.tolist()
    remaining_cost = 0.0
    for i in range(len(remaining)):
        node = remaining[i]
        # Find nearest among other unvisited + destination
        candidates = [d for d in remaining if d != node] + [destination_node]
        min_dist = min(distance_matrix[node, c] for c in candidates)
        remaining_cost += min_dist
    
    # Add cost to go from current to first unvisited
    first_unvisited = remaining[0]
    remaining_cost += distance_matrix[current_node, first_unvisited]
    
    return remaining_cost

def _beam_select(current_node, destination_node, unvisited_nodes, distance_matrix):
    """Beam search selection with pruning."""
    if _is_time_up():
        # Fall back to nearest neighbor for speed
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    beam_width = _beam_state['beam_width']
    
    # Generate candidates: all unvisited nodes
    candidates = unvisited_nodes.tolist()
    n_candidates = len(candidates)
    
    if n_candidates <= beam_width:
        # If few candidates, evaluate all
        scored = []
        for node in candidates:
            direct_cost = distance_matrix[current_node, node]
            # Estimate remaining cost
            new_unvisited = [n for n in candidates if n != node]
            est_remaining = _estimate_remaining_cost(node, destination_node, 
                                                     np.array(new_unvisited) if new_unvisited else np.array([]), 
                                                     distance_matrix)
            total_est = direct_cost + est_remaining
            scored.append((total_est, node))
        scored.sort(key=lambda x: x[0])
        return scored[0][1]
    
    # For larger candidate sets, use a two-stage approach:
    # 1. Take top-K candidates by direct distance
    # 2. Evaluate those more carefully
    direct_dists = distance_matrix[current_node, unvisited_nodes]
    top_k_indices = np.argpartition(direct_dists, min(beam_width * 2, n_candidates - 1))[:min(beam_width * 2, n_candidates)]
    
    scored = []
    for idx in top_k_indices:
        node = int(unvisited_nodes[idx])
        direct_cost = distance_matrix[current_node, node]
        new_unvisited = [int(n) for n in unvisited_nodes if n != node]
        est_remaining = _estimate_remaining_cost(node, destination_node, 
                                                 np.array(new_unvisited) if new_unvisited else np.array([]), 
                                                 distance_matrix)
        total_est = direct_cost + est_remaining
        scored.append((total_est, node))
    
    scored.sort(key=lambda x: x[0])
    return scored[0][1]

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Initialize state if this is the first call (tour just started)
    if not _beam_state['tours']:
        _init_beam_state()
    
    # If time is up, fall back to nearest neighbor
    if _is_time_up():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # Use beam search selection
    return _beam_select(current_node, destination_node, unvisited_nodes, distance_matrix)
# EVOLVE-BLOCK-END
