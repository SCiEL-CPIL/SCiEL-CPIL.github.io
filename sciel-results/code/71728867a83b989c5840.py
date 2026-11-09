# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 120.0
_SOFT_DEADLINE = 118.0

_state = {
    'n': 0,
    'step': 0,
    'instance_key': None
}

def _reset_state(n):
    _state['n'] = n
    _state['step'] = 0
    _state['instance_key'] = None

def _get_instance_key(distance_matrix):
    n = distance_matrix.shape[0]
    # Use a cheap fingerprint: sum of first row
    key = float(np.sum(distance_matrix[0, :min(10, n)]))
    return (n, key)

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Phase-aware constructive heuristic:
    - Early phase: pure Nearest Neighbor (simple, fast, avoids over-thinking early edges)
    - Late phase: destination-aware scoring to avoid long closing edge
    """
    # Deadline guard
    if time.time() > _SOFT_DEADLINE:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return current_node
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    n = distance_matrix.shape[0]
    
    # Detect new instance
    key = _get_instance_key(distance_matrix)
    if _state['instance_key'] != key:
        _reset_state(n)
        _state['instance_key'] = key
        _state['step'] = 0
    
    # Phase determination: switch to destination-aware in the last 20% of nodes
    phase_threshold = int(0.2 * n)
    is_late_phase = (n_unvisited <= phase_threshold)
    
    if is_late_phase:
        # Destination-aware scoring: minimize dist(current, u) + alpha * dist(u, destination)
        dist_from_current = distance_matrix[current_node, unvisited_nodes]
        dist_to_dest = distance_matrix[destination_node, unvisited_nodes]
        
        # Weight destination proximity more heavily as we get closer to the end
        remaining_frac = n_unvisited / n
        alpha = 1.0 / (remaining_frac + 0.1)  # alpha increases as we approach the end
        
        scores = dist_from_current + alpha * dist_to_dest
        idx = np.argmin(scores)
    else:
        # Early phase: pure Nearest Neighbor
        dist_from_current = distance_matrix[current_node, unvisited_nodes]
        idx = np.argmin(dist_from_current)
    
    _state['step'] += 1
    return int(unvisited_nodes[idx])
# EVOLVE-BLOCK-END
