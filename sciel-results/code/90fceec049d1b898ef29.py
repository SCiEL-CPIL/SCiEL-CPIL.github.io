"""Traveling salesman constructive heuristic: Geometric convex hull peeling with area penalty."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
_DEADLINE = 145.0  # Safety margin before 150s kill


def _elapsed():
    if _start_time is None:
        return 0.0
    return time.time() - _start_time


def _deadline_exceeded():
    return _elapsed() > _DEADLINE


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return int(current_node)
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    if _deadline_exceeded():
        # Fallback: nearest neighbor
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])

    # Get coordinates from distance matrix (Euclidean)
    # We need to extract 2D coordinates from the distance matrix.
    # Use classical MDS or simple approach: pick first two points as reference.
    # For Euclidean TSP, we can reconstruct coordinates using the first 3 points.
    # Actually, for 2D Euclidean, we can use the first point as origin, second on x-axis.
    
    n = distance_matrix.shape[0]
    
    # Reconstruct 2D coordinates from distance matrix
    # Use point 0 as origin, point 1 on positive x-axis
    # For any point k: 
    #   x_k = (d(0,k)^2 + d(0,1)^2 - d(k,1)^2) / (2 * d(0,1))
    #   y_k = sqrt(d(0,k)^2 - x_k^2)  (sign determined by orientation)
    
    d01 = distance_matrix[0, 1]
    if d01 < 1e-10:
        # Degenerate case, use nearest neighbor
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
    
    # Compute coordinates for all nodes
    coords = np.zeros((n, 2))
    coords[0, 0] = 0.0
    coords[0, 1] = 0.0
    coords[1, 0] = d01
    coords[1, 1] = 0.0
    
    # Determine sign convention: use point 2 to determine which side
    d02 = distance_matrix[0, 2]
    d12 = distance_matrix[1, 2]
    x2 = (d02**2 + d01**2 - d12**2) / (2.0 * d01)
    y2_sq = d02**2 - x2**2
    y2 = np.sqrt(max(0.0, y2_sq))
    # Assume point 2 is in upper half-plane (y > 0)
    coords[2, 0] = x2
    coords[2, 1] = y2
    
    # For other points, compute x, then determine y sign based on cross product with (0,1,2)
    for k in range(3, n):
        d0k = distance_matrix[0, k]
        d1k = distance_matrix[1, k]
        xk = (d0k**2 + d01**2 - d1k**2) / (2.0 * d01)
        yk_sq = d0k**2 - xk**2
        yk = np.sqrt(max(0.0, yk_sq))
        # Determine sign: cross product of (p1-p0) x (pk-p0) should have same sign as (p1-p0) x (p2-p0)
        # p1-p0 = (d01, 0), pk-p0 = (xk, yk)
        # cross = d01 * yk - 0 * xk = d01 * yk
        # p2-p0 = (x2, y2), cross_ref = d01 * y2
        # So sign of yk should match sign of y2 (which is positive)
        # But we need to be more careful: use the orientation of triangle (0,1,k) vs (0,1,2)
        # Actually, let's just use the sign that makes the point consistent with the bulk
        # For simplicity, assume all points are in the same half-plane relative to line 0-1
        # This is a simplification but works for convex-ish distributions
        coords[k, 0] = xk
        coords[k, 1] = yk
    
    # Now select next node using geometric heuristic
    # Cost = distance to current + alpha * area of triangle (current, destination, candidate)
    # The area term penalizes moves that swing away from the "perimeter"
    
    cur = coords[current_node]
    dest = coords[destination_node]
    
    # Get coordinates of unvisited nodes
    unv_idx = unvisited_nodes
    unv_coords = coords[unv_idx]
    
    # Compute distances from current to each unvisited
    dists = np.sqrt(np.sum((unv_coords - cur)**2, axis=1))
    
    # Compute area of triangle (current, destination, candidate) for each candidate
    # Area = 0.5 * |cross(dest - cur, cand - cur)|
    diff_dest = dest - cur
    diff_cands = unv_coords - cur
    cross = diff_dest[0] * diff_cands[:, 1] - diff_dest[1] * diff_cands[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Normalize areas to be comparable to distances
    max_dist = np.max(dists) if len(dists) > 0 else 1.0
    if max_dist < 1e-10:
        max_dist = 1.0
    norm_areas = areas / max_dist
    
    # Cost function: distance + alpha * normalized area
    # alpha controls the strength of the geometric penalty
    alpha = 0.5
    
    costs = dists + alpha * norm_areas
    
    best_idx = np.argmin(costs)
    return int(unv_idx[best_idx])
# EVOLVE-BLOCK-END
