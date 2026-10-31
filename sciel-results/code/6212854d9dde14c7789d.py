"""TSP constructive heuristic using potential field with spatial coverage."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state to persist across calls within a single tour construction
_state = {
    'start_node': None,
    'visited': None,
    'coords': None,
    'deadline': None,
    'tour': None,
    'call_count': 0,
}


def _reset_state(start_node, distance_matrix):
    _state['start_node'] = start_node
    _state['visited'] = set()
    _state['visited'].add(start_node)
    _state['coords'] = None
    _state['deadline'] = time.time() + 145.0  # Safety margin before 150s limit
    _state['tour'] = [start_node]
    _state['call_count'] = 0


def _is_deadline_passed():
    """Check if we are close to the time limit."""
    return time.time() > _state['deadline']


def _get_coords(distance_matrix):
    """Recover approximate 2D coordinates from distance matrix."""
    if _state['coords'] is not None:
        return _state['coords']
    
    n = distance_matrix.shape[0]
    if n < 2:
        _state['coords'] = np.zeros((n, 2))
        return _state['coords']
    
    # Use simple 2D embedding via distances to first two points
    # Place point 0 at origin, point 1 on x-axis
    coords = np.zeros((n, 2))
    d01 = distance_matrix[0, 1]
    coords[1, 0] = d01
    
    for i in range(2, n):
        d0i = distance_matrix[0, i]
        d1i = distance_matrix[1, i]
        # Solve for xi, yi
        xi = (d0i**2 - d1i**2 + d01**2) / (2 * d01) if d01 > 1e-10 else 0.0
        yi_sq = d0i**2 - xi**2
        yi = np.sqrt(max(yi_sq, 0.0))
        coords[i, 0] = xi
        coords[i, 1] = yi
    
    # Try to fix signs using a third reference if available
    if n >= 3:
        ref = 2
        for i in range(3, n):
            # Check consistency with distance to ref
            d_ref_i = distance_matrix[ref, i]
            # Current distance
            dx = coords[i, 0] - coords[ref, 0]
            dy = coords[i, 1] - coords[ref, 1]
            curr_dist = np.sqrt(dx*dx + dy*dy)
            # Try flipping y sign
            dy_flipped = -coords[i, 1] + coords[ref, 1]
            dist_flipped = np.sqrt(dx*dx + dy_flipped*dy_flipped)
            if abs(dist_flipped - d_ref_i) < abs(curr_dist - d_ref_i):
                coords[i, 1] = -coords[i, 1]
    
    _state['coords'] = coords
    return _state['coords']


def _potential_field_score(current_node, unvisited_nodes, distance_matrix, coords):
    """
    Compute potential field score for each candidate.
    Score = attraction to current node + repulsion from other unvisited nodes.
    """
    n_cand = len(unvisited_nodes)
    if n_cand == 0:
        return np.array([])
    
    current_pos = coords[current_node]
    cand_positions = coords[unvisited_nodes]
    
    # Attraction: distance from current to each candidate
    attractions = np.linalg.norm(cand_positions - current_pos, axis=1)
    
    # Repulsion: sum of inverse distances to all other unvisited nodes
    # This encourages spreading out and prevents clustering
    repulsions = np.zeros(n_cand)
    for i in range(n_cand):
        # Distance from candidate i to all other unvisited nodes
        diffs = cand_positions - cand_positions[i]
        dists = np.linalg.norm(diffs, axis=1)
        # Avoid division by zero (distance to self is 0)
        dists[i] = 1e10
        # Inverse distance repulsion (capped to avoid extreme values)
        inv_dists = 1.0 / np.maximum(dists, 0.01)
        repulsions[i] = np.sum(inv_dists)
    
    # Normalize repulsion to be comparable to attraction
    max_attr = np.max(attractions) if n_cand > 0 else 1.0
    max_rep = np.max(repulsions) if n_cand > 0 else 1.0
    
    # Combine: primarily attraction (go to nearby nodes), secondarily repulsion (spread out)
    # Weight: 0.8 attraction, 0.2 repulsion
    weight_attr = 0.8
    weight_rep = 0.2
    
    score = weight_attr * (attractions / max_attr if max_attr > 1e-10 else attractions) + \
            weight_rep * (repulsions / max_rep if max_rep > 1e-10 else repulsions)
    
    return score


def _nearest_neighbor_score(current_node, unvisited_nodes, distance_matrix):
    """Simple nearest neighbor score."""
    if len(unvisited_nodes) == 0:
        return np.array([])
    return distance_matrix[current_node, unvisited_nodes]


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Initialize state if this is the first call
    if _state['start_node'] is None:
        _reset_state(current_node, distance_matrix)
    
    _state['call_count'] += 1
    
    # Check deadline
    if _is_deadline_passed():
        # Fallback to nearest neighbor if over time
        if len(unvisited_nodes) == 0:
            return int(current_node)
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    if len(unvisited_nodes) == 0:
        return int(current_node)
    
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # Get coordinates
    coords = _get_coords(distance_matrix)
    
    # For the first few steps, use pure nearest neighbor to establish a good initial path
    if _state['call_count'] <= 3:
        scores = _nearest_neighbor_score(current_node, unvisited_nodes, distance_matrix)
        best_idx = np.argmin(scores)
        next_node = int(unvisited_nodes[best_idx])
        _state['visited'].add(next_node)
        _state['tour'].append(next_node)
        return next_node
    
    # For later steps, use potential field approach
    scores = _potential_field_score(current_node, unvisited_nodes, distance_matrix, coords)
    best_idx = np.argmin(scores)
    next_node = int(unvisited_nodes[best_idx])
    
    # Track visited
    _state['visited'].add(next_node)
    _state['tour'].append(next_node)
    
    return next_node
# EVOLVE-BLOCK-END
