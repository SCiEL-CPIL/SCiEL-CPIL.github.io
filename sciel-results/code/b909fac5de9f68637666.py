"""Direct heuristic for TSP next-node selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0  # safety margin before 150s hard kill

# Cache for recovered coordinates
_coord_cache = {}


def _recover_coords(distance_matrix, max_dim=2):
    """Recover 2D coordinates from distance matrix using classical MDS."""
    key = id(distance_matrix)
    if key in _coord_cache:
        return _coord_cache[key]
    
    d = distance_matrix
    n = d.shape[0]
    
    # Classical MDS
    D2 = d ** 2
    # Double centering
    ones = np.ones((n, n)) / (2 * n)
    row_sum = D2.sum(axis=1, keepdims=True) / (2 * n)
    col_sum = D2.sum(axis=0, keepdims=True) / (2 * n)
    grand_mean = D2.sum() / (2 * n * n)
    
    B = -0.5 * (D2 - row_sum - col_sum.T + grand_mean)
    
    # Eigen-decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    # Sort by descending eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Take top 2 dimensions
    top_eigvals = np.sqrt(np.maximum(eigenvalues[:max_dim], 0))
    coords = eigenvectors[:, :max_dim] * top_eigvals
    
    _coord_cache[key] = coords
    return coords


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using convex hull peeling / angular sweep heuristic.
    
    Key insight: nearest-neighbor creates edge crossings. Instead, we trace
    the convex hull of unvisited points first, then fill interior points.
    This produces a much smoother tour with fewer crossings.
    """
    # Time guard - if we're running out of time, fall back to greedy
    if time.time() - _START_TIME > _DEADLINE:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    if n_unvisited == 2:
        # For two nodes left, pick the one that minimizes total edge cost to close the tour
        u, v = unvisited_nodes
        cost_u = distance_matrix[current_node, u] + distance_matrix[u, v] + distance_matrix[v, destination_node]
        cost_v = distance_matrix[current_node, v] + distance_matrix[v, u] + distance_matrix[u, destination_node]
        return int(unvisited_nodes[0] if cost_u <= cost_v else unvisited_nodes[1])
    if n_unvisited == 3:
        # For three nodes, evaluate all 6 permutations
        u, v, w = unvisited_nodes
        perms = [
            (u, v, w), (u, w, v),
            (v, u, w), (v, w, u),
            (w, u, v), (w, v, u)
        ]
        best_cost = float('inf')
        best_first = u
        for p in perms:
            c = (distance_matrix[current_node, p[0]] + 
                 distance_matrix[p[0], p[1]] + 
                 distance_matrix[p[1], p[2]] + 
                 distance_matrix[p[2], destination_node])
            if c < best_cost:
                best_cost = c
                best_first = p[0]
        return int(best_first)
    
    # Recover coordinates for geometric heuristic
    try:
        coords = _recover_coords(distance_matrix)
    except Exception:
        # Fallback to distance-based if MDS fails
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # Get coordinates of unvisited nodes and current/destination
    curr_coords = coords[current_node]
    dest_coords = coords[destination_node]
    unvisited_coords = coords[unvisited_nodes]
    
    # Compute angles from current node to each unvisited node
    vecs = unvisited_coords - curr_coords
    norms = np.linalg.norm(vecs, axis=1)
    # Avoid division by zero
    norms[norms == 0] = 1e-10
    unit_vecs = vecs / norms[:, np.newaxis]
    
    # Angle of each vector relative to x-axis
    angles = np.arctan2(unit_vecs[:, 1], unit_vecs[:, 0])
    
    # Sort unvisited nodes by angle around current node
    angle_order = np.argsort(angles)
    
    # Now we want to find the next node that continues the "sweep" direction.
    # The idea: if we've been going in a certain direction, continue in that direction.
    # We approximate the "current direction" as the direction from current to destination
    # blended with the direction of the previous edge (which we don't have directly).
    # 
    # Better approach: compute the convex hull of unvisited points, and prefer
    # points on or near the convex hull, ordered by angle.
    
    # Compute convex hull of unvisited points
    # Use a simple gift-wrapping approach for small sets, or scipy if available
    try:
        from scipy.spatial import ConvexHull
        if n_unvisited >= 3:
            hull = ConvexHull(unvisited_coords)
            hull_points = set(hull.vertices.tolist())
        else:
            hull_points = set(range(n_unvisited))
    except Exception:
        # Fallback: all points considered "on hull"
        hull_points = set(range(n_unvisited))
    
    # Score each unvisited node:
    # 1. Prefer nodes on the convex hull (lower score = better)
    # 2. Among hull nodes, prefer those that continue the angular sweep
    # 3. Among non-hull nodes, prefer those closer to current (for filling interior)
    
    # Determine the "sweep direction" - we use the direction from current toward destination
    # as a global guide, but primarily follow the angular ordering
    
    # The key insight: we want to trace the hull in angular order.
    # Find which hull point is "next" in the angular sweep from the current position.
    
    # Compute the angle from current to destination as a reference
    dest_vec = dest_coords - curr_coords
    dest_norm = np.linalg.norm(dest_vec)
    if dest_norm > 1e-10:
        dest_angle = np.arctan2(dest_vec[1], dest_vec[0])
    else:
        dest_angle = 0.0
    
    # For each unvisited node, compute a composite score:
    # - Hull bonus: prefer hull points
    # - Angular continuity: prefer points that are close in angle to the "expected next" angle
    # - Distance penalty: prefer closer points (scaled down for hull points)
    
    # The "expected next angle" is approximated by the angle of the nearest hull point
    # that is ahead of us in the sweep. For simplicity, we use the destination angle
    # as a guide and blend with angular ordering.
    
    # Compute angular distance from each point to the destination direction
    # (wrapped to [-pi, pi])
    ang_diffs = np.abs((angles - dest_angle + np.pi) % (2 * np.pi) - np.pi)
    
    # Hull indicator: 0 for hull points, 1 for interior
    hull_ind = np.array([1.0 if i not in hull_points else 0.0 for i in range(n_unvisited)])
    
    # Distance from current
    dists = norms
    
    # Normalize distances
    max_dist = np.max(dists) if np.max(dists) > 0 else 1.0
    dist_norm = dists / max_dist
    
    # Composite score:
    # - Strong preference for hull points (weight 2.0)
    # - Moderate preference for angular alignment with destination (weight 0.5)
    # - Mild preference for closer points (weight 0.3)
    scores = 2.0 * hull_ind + 0.5 * (ang_diffs / np.pi) + 0.3 * dist_norm
    
    # For interior points, we want to fill them in when the hull is mostly done.
    # As the number of unvisited points decreases, we shift weight toward distance.
    progress = 1.0 - n_unvisited / 50.0
    if progress > 0.5:
        # More emphasis on distance as we fill interior
        interior_weight = 0.5 + 0.5 * (progress - 0.5)
        scores = (1 - interior_weight) * scores + interior_weight * dist_norm
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
