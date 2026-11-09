"""Geometric angle-based TSP constructive heuristic with convex-hull awareness."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track global state to detect when a new instance starts
_global_state = {
    'instance_id': -1,
    'call_count': 0,
    'start_time': None,
    'n': 0,
    'coords': None,
    'visited_order': [],
    'current_instance_hash': None
}


def _reset_state(distance_matrix):
    """Reset state when a new instance is detected."""
    n = distance_matrix.shape[0]
    if _global_state['n'] != n or _global_state['instance_id'] == -1:
        _global_state['instance_id'] += 1
        _global_state['call_count'] = 0
        _global_state['start_time'] = time.time()
        _global_state['n'] = n
        _global_state['coords'] = None
        _global_state['visited_order'] = []
        _global_state['current_instance_hash'] = None


def _get_coords(distance_matrix):
    """
    Try to recover 2D coordinates from the distance matrix using MDS.
    If not already computed, compute once per instance.
    """
    if _global_state['coords'] is not None:
        return _global_state['coords']
    
    n = distance_matrix.shape[0]
    
    # Use classical MDS to recover coordinates
    # D is the distance matrix
    D = distance_matrix
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    
    # Get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    
    # Sort by eigenvalue descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Take top 2 dimensions
    if eigenvalues.shape[0] >= 2:
        lam = np.sqrt(np.maximum(eigenvalues[:2], 0))
        coords = eigenvectors[:, :2] * lam
    else:
        coords = np.zeros((n, 2))
    
    _global_state['coords'] = coords
    return coords


def _compute_convex_hull_order(coords):
    """
    Compute the order of points along the convex hull using a simple
    angle-sorting approach around the centroid.
    Returns a set of node indices that are on the convex hull.
    """
    n = len(coords)
    if n <= 3:
        return set(range(n))
    
    centroid = coords.mean(axis=0)
    angles = np.arctan2(coords[:, 1] - centroid[1], coords[:, 0] - centroid[0])
    sorted_indices = np.argsort(angles)
    
    # Use Graham scan to find convex hull
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    # Sort by angle, then by distance
    sorted_pts = [(coords[i], i) for i in sorted_indices]
    
    # Lower hull
    lower = []
    for p, idx in sorted_pts:
        while len(lower) >= 2 and cross(lower[-2][0], lower[-1][0], p) <= 0:
            lower.pop()
        lower.append((p, idx))
    
    # Upper hull
    upper = []
    for p, idx in reversed(sorted_pts):
        while len(upper) >= 2 and cross(upper[-2][0], upper[-1][0], p) <= 0:
            upper.pop()
        upper.append((p, idx))
    
    hull_indices = set()
    for p, idx in lower[:-1]:
        hull_indices.add(idx)
    for p, idx in upper[:-1]:
        hull_indices.add(idx)
    
    return hull_indices


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a geometric angle-based heuristic.
    Prioritizes nodes that maintain a consistent direction of travel,
    with a bias toward convex hull nodes to reduce edge crossings.
    """
    _reset_state(distance_matrix)
    
    # Deadline guard
    if _global_state['start_time'] is not None:
        elapsed = time.time() - _global_state['start_time']
        if elapsed > 110:  # Safety margin before 120s limit
            # Fallback to nearest neighbor
            return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n = distance_matrix.shape[0]
    remaining = len(unvisited_nodes)
    
    # Get coordinates
    coords = _get_coords(distance_matrix)
    
    # Compute convex hull once per instance (cache it)
    if not hasattr(_global_state, 'hull_set') or _global_state.get('hull_set') is None:
        _global_state['hull_set'] = _compute_convex_hull_order(coords)
    hull_set = _global_state['hull_set']
    
    # Track visited order for direction computation
    visited_order = _global_state['visited_order']
    visited_order.append(current_node)
    
    # If this is the first pick, just use nearest neighbor
    if len(visited_order) <= 1:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # Compute the incoming direction vector
    if len(visited_order) >= 2:
        prev_node = visited_order[-2]
        dir_vec = coords[current_node] - coords[prev_node]
        dir_norm = np.linalg.norm(dir_vec)
        if dir_norm < 1e-10:
            dir_vec = np.array([1.0, 0.0])
        else:
            dir_vec = dir_vec / dir_norm
    else:
        dir_vec = np.array([1.0, 0.0])
    
    # Score each unvisited node
    scores = np.zeros(len(unvisited_nodes))
    
    for i, node in enumerate(unvisited_nodes):
        # Distance to current node
        dist_current = distance_matrix[current_node, node]
        
        # Distance to destination (to help close the tour)
        dist_dest = distance_matrix[node, destination_node]
        
        # Angle deviation: how much does the direction to this node deviate from current direction
        vec_to_node = coords[node] - coords[current_node]
        vec_norm = np.linalg.norm(vec_to_node)
        if vec_norm < 1e-10:
            angle_dev = np.pi  # Max deviation
        else:
            vec_to_node = vec_to_node / vec_norm
            cos_angle = np.clip(np.dot(dir_vec, vec_to_node), -1, 1)
            angle_dev = np.arccos(cos_angle)
        
        # Hull bonus: prefer convex hull nodes (they should be visited in order)
        hull_bonus = 0.0
        if node in hull_set:
            hull_bonus = -0.15 * dist_current  # Slight preference for hull nodes
        
        # Phase-aware weighting: blend angle and distance based on progress
        progress = 1.0 - (remaining / n)
        
        # Early: more weight on angle (to trace hull), Late: more weight on distance to destination
        if progress < 0.3:
            # Early phase: prioritize angle consistency and hull
            weight_angle = 1.0
            weight_dist_current = 0.5
            weight_dist_dest = 0.0
        elif progress < 0.7:
            # Middle phase: balance
            weight_angle = 0.7
            weight_dist_current = 0.6
            weight_dist_dest = 0.2
        else:
            # Late phase: prioritize closing the tour
            weight_angle = 0.3
            weight_dist_current = 0.5
            weight_dist_dest = 0.8
        
        # Normalize distances for fair comparison
        max_dist = np.max(distance_matrix) if n > 1 else 1.0
        norm_dist_current = dist_current / max_dist
        norm_dist_dest = dist_dest / max_dist
        norm_angle = angle_dev / np.pi
        
        score = (weight_angle * norm_angle + 
                 weight_dist_current * norm_dist_current + 
                 weight_dist_dest * norm_dist_dest)
        
        # Apply hull bonus
        score += hull_bonus / max_dist if max_dist > 0 else 0
        
        scores[i] = score
    
    # Select the node with minimum score
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
