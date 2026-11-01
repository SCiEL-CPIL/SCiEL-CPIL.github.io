import numpy as np
import time
import itertools

N = 14

def _triangle_areas(pts):
    """Compute all triangle areas for N points. Returns array of shape (N, N, N) or flattened."""
    n = len(pts)
    areas = np.zeros((n, n, n))
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = pts[i]
                b = pts[j]
                c = pts[k]
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
                areas[i, j, k] = area
                areas[j, i, k] = area
                areas[j, k, i] = area
                areas[k, i, j] = area
                areas[k, j, i] = area
                areas[i, k, j] = area
    return areas

def _min_triangle_area(pts):
    """Return the minimum triangle area among all triples."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = pts[i]
                b = pts[j]
                c = pts[k]
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(pts):
    """Compute the area of the convex hull of pts."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: use shoelace formula on sorted points
        # This is a rough fallback
        return 1.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    min_area = _min_triangle_area(pts)
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-12:
        return 0.0
    return min_area / hull_area

def _soft_min_triangle_area(pts, beta=100.0):
    """Soft minimum of triangle areas using log-sum-exp trick."""
    n = len(pts)
    log_sum = 0.0
    min_val = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = pts[i]
                b = pts[j]
                c = pts[k]
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
                if area < min_val:
                    min_val = area
                if area > 0:
                    log_sum += np.exp(-beta * area)
    if log_sum > 0:
        return -np.log(log_sum) / beta
    return min_val

def _regular_polygon(n, radius=1.0):
    """Generate vertices of a regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))

def _coordinate_descent(pts, max_iters=50, step_size=0.01, beta=100.0):
    """Perform coordinate descent to improve the configuration."""
    n = len(pts)
    best_pts = pts.copy()
    best_score = _score(best_pts)
    
    for iteration in range(max_iters):
        improved = False
        for i in range(n):
            orig = best_pts[i].copy()
            # Try small perturbations in x and y directions
            for dx in [-step_size, 0, step_size]:
                for dy in [-step_size, 0, step_size]:
                    if dx == 0 and dy == 0:
                        continue
                    new_pts = best_pts.copy()
                    new_pts[i] = orig + np.array([dx, dy])
                    new_score = _score(new_pts)
                    if new_score > best_score + 1e-10:
                        best_pts = new_pts
                        best_score = new_score
                        improved = True
        if not improved:
            # Try smaller steps
            step_size *= 0.5
            if step_size < 1e-6:
                break
    
    return best_pts, best_score

def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Start with regular polygon
    pts = _regular_polygon(N)
    
    # Normalize to unit area convex hull
    hull_area = _convex_hull_area(pts)
    if hull_area > 0:
        pts = pts * np.sqrt(1.0 / hull_area)
    
    best_pts = pts.copy()
    best_score = _score(best_pts)
    
    # Coordinate descent refinement
    if time.time() < deadline - 5:
        best_pts, best_score = _coordinate_descent(best_pts, max_iters=30, step_size=0.05)
    
    # Try different starting configurations if time permits
    if time.time() < deadline - 10:
        # Slightly perturbed regular polygon
        rng = np.random.RandomState(42)
        for trial in range(5):
            if time.time() > deadline - 5:
                break
            noise = rng.randn(N, 2) * 0.1
            trial_pts = _regular_polygon(N) + noise
            hull_area = _convex_hull_area(trial_pts)
            if hull_area > 0:
                trial_pts = trial_pts * np.sqrt(1.0 / hull_area)
            trial_pts, trial_score = _coordinate_descent(trial_pts, max_iters=20, step_size=0.05)
            if trial_score > best_score:
                best_pts = trial_pts
                best_score = trial_score
    
    # Final polish with smaller steps if time permits
    if time.time() < deadline - 5:
        best_pts, best_score = _coordinate_descent(best_pts, max_iters=20, step_size=0.01)
    
    return best_pts
