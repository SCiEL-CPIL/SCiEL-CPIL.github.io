"""Optimise 50 equal circles inside the unit circle: maximise the common radius.

Score = min( min_{i<j} |c_i - c_j| / 2 ,  min_i (1 - |c_i|) ).
Strategy (Voronoi/Delaunay geometric optimization):
  * Build Delaunay triangulation of current point set
  * Compute Voronoi cells and their inradii
  * Use geometric gradient: move each point away from its closest constraint
    (either a neighboring circle or the container boundary)
  * Multi-stage annealing with increasing precision
  * Deterministic multi-restart from diverse ring configurations
  * Explicit wall-clock deadline guard
"""
import time
import numpy as np
from scipy.spatial import Delaunay, Voronoi
from scipy.optimize import minimize

N = 50
T_LIMIT = 120.0
T_SAFE = 105.0
T_START = time.time()


def _score(points):
    """Exact task score in double precision (or -inf if invalid)."""
    if points is None or points.shape != (N, 2):
        return -np.inf
    if not np.all(np.isfinite(points)):
        return -np.inf
    d = 1.0 - np.linalg.norm(points, axis=1)
    if np.any(d < -1e-12):
        return -np.inf
    p = points
    diff = p[:, None, :] - p[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(N, 1)
    mn = np.min(dist[iu]) / 2.0
    return float(min(mn, np.min(d)))


def _voronoi_inradii(points):
    """Compute inradius of each Voronoi cell (distance to nearest edge or boundary).
    
    Returns array of shape (N,) where inradii[i] is the maximum radius of a circle
    centered at points[i] that fits within the Voronoi cell and the unit disk.
    """
    try:
        vor = Voronoi(points)
    except Exception:
        # Fallback: use pairwise distances
        diff = points[:, None, :] - points[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dist, np.inf)
        pair_min = np.min(dist, axis=1) / 2.0
        boundary = 1.0 - np.linalg.norm(points, axis=1)
        return np.minimum(pair_min, boundary)
    
    inradii = np.empty(N)
    for i in range(N):
        # Get vertices of Voronoi cell i
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        
        if len(region) == 0:
            # Point is on convex hull, cell is unbounded
            # Inradius is limited by distance to boundary and nearest neighbor
            diff = points[i] - points
            dist = np.sqrt(np.sum(diff * diff, axis=1))
            np.fill_diagonal(dist, np.inf)
            nearest = np.min(dist) / 2.0
            boundary = 1.0 - np.linalg.norm(points[i])
            inradii[i] = min(nearest, boundary)
            continue
        
        # Compute distance from point to each edge of the Voronoi cell
        min_dist = np.inf
        for j in range(len(region)):
            v1 = vor.vertices[region[j]]
            v2 = vor.vertices[region[(j + 1) % len(region)]]
            # Distance from points[i] to line segment v1-v2
            d_vec = v2 - v1
            d_len = np.linalg.norm(d_vec)
            if d_len < 1e-15:
                continue
            t = np.clip(np.dot(points[i] - v1, d_vec) / (d_len * d_len), 0, 1)
            proj = v1 + t * d_vec
            dist_to_edge = np.linalg.norm(points[i] - proj)
            min_dist = min(min_dist, dist_to_edge)
        
        # Also consider distance to unit circle boundary
        boundary = 1.0 - np.linalg.norm(points[i])
        inradii[i] = min(min_dist, boundary)
    
    return inradii


def _geometric_gradient(points, learning_rate=0.01):
    """Compute geometric gradient for Voronoi-based optimization.
    
    Each point is pushed away from its most constraining neighbor/boundary.
    Returns gradient of shape (N, 2) to be added to points.
    """
    n = len(points)
    grad = np.zeros_like(points)
    
    # Compute pairwise distances
    diff = points[:, None, :] - points[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # Compute boundary distances
    boundary_dist = 1.0 - np.linalg.norm(points, axis=1)
    
    # For each point, find its most constraining constraint
    for i in range(n):
        # Find closest neighbor
        nearest_j = np.argmin(dist[i])
        d_nn = dist[i, nearest_j]
        
        # Find boundary constraint
        d_bound = boundary_dist[i]
        
        # Determine which is more constraining
        if d_nn / 2.0 < d_bound:
            # Neighbor is more constraining: push away from nearest neighbor
            direction = points[i] - points[nearest_j]
            norm = np.linalg.norm(direction)
            if norm > 1e-15:
                direction /= norm
                # Push away with strength proportional to violation
                violation = max(0, 1.0 - d_nn)
                grad[i] += direction * violation * learning_rate
        else:
            # Boundary is more constraining: push inward
            norm = np.linalg.norm(points[i])
            if norm > 1e-15:
                direction = -points[i] / norm
                violation = max(0, d_bound - 0.05)  # small buffer
                grad[i] += direction * violation * learning_rate
    
    return grad


def _voronoi_optimize(points, max_iter=200, lr_schedule='linear'):
    """Optimize points using Voronoi-based geometric gradient.
    
    Args:
        points: initial positions (N, 2)
        max_iter: maximum iterations
        lr_schedule: 'linear' or 'exponential'
    
    Returns:
        optimized points (N, 2)
    """
    current = points.copy()
    best_score = _score(current)
    best_points = current.copy()
    
    for it in range(max_iter):
        if time.time() - T_START > T_SAFE:
            break
        
        # Adaptive learning rate
        if lr_schedule == 'linear':
            lr = 0.05 * (1.0 - it / max_iter) + 0.001
        else:
            lr = 0.05 * (0.95 ** it) + 0.001
        
        # Compute geometric gradient
        grad = _geometric_gradient(current, learning_rate=1.0)
        
        # Update
        new_points = current + grad * lr
        
        # Validate
        new_score = _score(new_points)
        
        if new_score > best_score:
            best_score = new_score
            best_points = new_points.copy()
            current = new_points
        else:
            # If score decreased, try smaller step
            new_points_small = current + grad * lr * 0.5
            small_score = _score(new_points_small)
            if small_score > best_score:
                best_score = small_score
                best_points = new_points_small.copy()
                current = new_points_small
            else:
                # No improvement, reduce learning rate and continue
                if it > 50 and small_score < best_score * 0.99:
                    break
    
    return best_points


def _ring_guess(rings):
    """Deterministic multi-ring guess."""
    pts = [[0.0, 0.0]]
    for rad, cnt in rings:
        if cnt <= 0:
            continue
        a = np.arange(cnt) * 2.0 * np.pi / cnt
        pts.extend(np.c_[rad * np.cos(a), rad * np.sin(a)].tolist())
    return np.array(pts[:N])


def _hex_guess():
    """Hexagonal lattice guess clipped to unit disk."""
    pts = []
    # Generate hexagonal lattice points
    for i in range(-6, 7):
        for j in range(-6, 7):
            x = i + 0.5 * (j % 2)
            y = j * np.sqrt(3) / 2
            if x * x + y * y <= 0.95 * 0.95:
                pts.append([x * 0.15, y * 0.15])
    return np.array(pts[:N]) if len(pts) >= N else _ring_guess([(0.87, 49)])


def construct():
    global T_START
    T_START = time.time()
    
    # --- Diverse initial guesses ---
    guesses = [
        _ring_guess([(0.30, 12), (0.60, 12), (0.87, 25)]),
        _ring_guess([(0.34, 10), (0.65, 12), (0.87, 27)]),
        _ring_guess([(0.28, 10), (0.56, 12), (0.80, 14), (0.93, 13)]),
        _ring_guess([(0.32, 11), (0.62, 11), (0.90, 27)]),
        _ring_guess([(0.36, 12), (0.70, 13), (0.90, 24)]),
        _ring_guess([(0.30, 9), (0.60, 11), (0.85, 13), (0.95, 16)]),
        _hex_guess(),
        _ring_guess([(0.25, 8), (0.50, 12), (0.75, 16), (0.95, 13)]),
        _ring_guess([(0.35, 12), (0.70, 16), (0.92, 21)]),
    ]
    
    best_x, best_val = None, -np.inf
    for g in guesses:
        v = _score(g)
        if v > best_val:
            best_val, best_x = v, g.copy()
    
    # --- Stage 1: Coarse Voronoi optimization from all starts ---
    for g in guesses:
        if time.time() - T_START > T_SAFE:
            break
        x = _voronoi_optimize(g, max_iter=150, lr_schedule='linear')
        v = _score(x)
        if v > best_val:
            best_val, best_x = v, x.copy()
    
    # --- Stage 2: Fine optimization from best result ---
    if best_x is not None and time.time() - T_START < T_SAFE - 5.0:
        x = _voronoi_optimize(best_x, max_iter=300, lr_schedule='exponential')
        v = _score(x)
        if v > best_val:
            best_val, best_x = v, x.copy()
    
    # --- Stage 3: Very fine polish ---
    if best_x is not None and time.time() - T_START < T_SAFE - 3.0:
        x = _voronoi_optimize(best_x, max_iter=500, lr_schedule='exponential')
        v = _score(x)
        if v > best_val:
            best_val, best_x = v, x.copy()
    
    # --- Final validation and fallback ---
    out = best_x
    if out is None or out.shape != (N, 2) or not np.all(np.isfinite(out)) \
            or _score(out) < -1e-9:
        for g in guesses:
            if _score(g) > -1e-9:
                return g
        return _ring_guess([(0.87, 49)])
    return out
