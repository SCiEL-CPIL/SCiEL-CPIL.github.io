"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time
from itertools import combinations

def _min_triangle_area_vectorized(points):
    """Compute the minimum area over all triples of points using vectorized ops."""
    n = len(points)
    # Generate all index combinations
    idx = np.array(list(combinations(range(n), 3)))
    # Shape: (num_triples, 3, 2)
    tri = points[idx]  # (num_triples, 3, 2)
    # Compute areas: 0.5 * |cross(p1-p0, p2-p0)|
    p0 = tri[:, 0, :]
    p1 = tri[:, 1, :]
    p2 = tri[:, 2, :]
    cross = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    try:
        min_area = _min_triangle_area_vectorized(points)
        hull_area = _convex_hull_area(points)
        if hull_area <= 0 or not np.isfinite(min_area):
            return 0.0
        return min_area / hull_area
    except Exception:
        return 0.0

def _regular_polygon(n, rotation=0.0):
    """Generate a regular n-gon centered at origin."""
    angles = rotation + 2 * np.pi * np.arange(n) / n
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _two_rings(n_outer, n_inner, r_outer=1.0, r_inner=0.5, rot_outer=0.0, rot_inner=0.0):
    """Generate points on two concentric rings."""
    outer = (r_outer * _regular_polygon(n_outer, rot_outer))
    inner = (r_inner * _regular_polygon(n_inner, rot_inner))
    return np.vstack([outer, inner])

def _hexagonal_packing(n):
    """Generate points in a hexagonal-like pattern."""
    # Use a grid with hexagonal offset
    points = []
    # Try different grid arrangements
    for rows in range(1, int(np.sqrt(n)) + 3):
        for cols in range(1, int(np.sqrt(n)) + 3):
            if rows * cols >= n:
                # Generate grid
                xs = np.linspace(-1, 1, cols)
                ys = np.linspace(-1, 1, rows)
                grid = np.array([[x, y] for y in ys for x in xs])
                # Add hexagonal offset to odd rows
                for i in range(len(grid)):
                    if (i // cols) % 2 == 1:
                        grid[i, 0] += 0.3
                # Take first n points
                if len(grid) >= n:
                    return grid[:n]
    # Fallback to regular polygon
    return _regular_polygon(n)

def _optimize_local(points, max_iter=200, step_size=0.05):
    """Local optimization using coordinate descent with adaptive step size."""
    best_points = points.copy()
    best_score = _score(best_points)
    current_points = points.copy()
    
    for iteration in range(max_iter):
        improved = False
        # Try moving each point in each direction
        for i in range(N):
            for dx, dy in [(step_size, 0), (-step_size, 0), (0, step_size), (0, -step_size),
                          (step_size*0.7, step_size*0.7), (-step_size*0.7, step_size*0.7),
                          (step_size*0.7, -step_size*0.7), (-step_size*0.7, -step_size*0.7)]:
                candidate = current_points.copy()
                candidate[i, 0] += dx
                candidate[i, 1] += dy
                sc = _score(candidate)
                if sc > best_score + 1e-12:
                    best_score = sc
                    best_points = candidate.copy()
                    current_points = candidate.copy()
                    improved = True
                    break
            if improved:
                break
        if not improved:
            # Reduce step size
            step_size *= 0.7
            if step_size < 1e-6:
                break
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Leave 15s safety margin
    
    # Generate multiple starting configurations
    candidates = []
    
    # Regular 14-gon
    candidates.append(_regular_polygon(N))
    
    # Two rings: 8 outer + 6 inner
    candidates.append(_two_rings(8, 6, r_outer=1.0, r_inner=0.55, rot_inner=np.pi/6))
    
    # Two rings: 7 outer + 7 inner
    candidates.append(_two_rings(7, 7, r_outer=1.0, r_inner=0.5, rot_inner=np.pi/7))
    
    # Two rings: 9 outer + 5 inner
    candidates.append(_two_rings(9, 5, r_outer=1.0, r_inner=0.6, rot_inner=np.pi/9))
    
    # Two rings: 10 outer + 4 inner
    candidates.append(_two_rings(10, 4, r_outer=1.0, r_inner=0.65, rot_inner=np.pi/10))
    
    # Two rings: 6 outer + 8 inner
    candidates.append(_two_rings(6, 8, r_outer=1.0, r_inner=0.5, rot_inner=np.pi/8))
    
    # Hexagonal packing
    candidates.append(_hexagonal_packing(N))
    
    # Evaluate all candidates
    best_points = candidates[0].copy()
    best_score = _score(best_points)
    
    for cand in candidates[1:]:
        sc = _score(cand)
        if sc > best_score:
            best_score = sc
            best_points = cand.copy()
    
    # Local optimization from best starting point
    if time.time() < deadline - 30:
        best_points, best_score = _optimize_local(best_points, max_iter=300, step_size=0.08)
    
    # Try local optimization from other good starting points if time remains
    for cand in candidates:
        if time.time() > deadline - 15:
            break
        if np.allclose(cand, best_points, atol=1e-6):
            continue
        sc = _score(cand)
        if sc > best_score * 0.95:  # Only try if reasonably good
            opt_pts, opt_sc = _optimize_local(cand, max_iter=100, step_size=0.05)
            if opt_sc > best_score:
                best_score = opt_sc
                best_points = opt_pts.copy()
    
    # Final fine-tuning with smaller steps
    if time.time() < deadline - 10:
        best_points, best_score = _optimize_local(best_points, max_iter=200, step_size=0.02)
    
    return best_points
# EVOLVE-BLOCK-END
