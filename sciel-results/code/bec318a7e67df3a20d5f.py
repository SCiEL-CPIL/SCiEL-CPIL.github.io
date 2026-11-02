"""Heilbronn convex regions: maximize minimum triangle area / convex hull area for 13 points."""
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
import time

N = 13

# EVOLVE-BLOCK-START

def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: use a simple polygon area if hull fails
        if len(points) < 3:
            return 0.0
        # Sort by angle around centroid
        centroid = points.mean(axis=0)
        angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
        order = np.argsort(angles)
        pts = points[order]
        x, y = pts[:, 0], pts[:, 1]
        return 0.5 * np.abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def _min_triangle_area(points):
    """Compute the minimum area among all triples of points."""
    n = len(points)
    min_area = float('inf')
    # Precompute all cross products
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross(pj - pi, pk - pi)|
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
                if cross < min_area:
                    min_area = cross
    return 0.5 * min_area


def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area


def _regular_polygon(n, radius=1.0, center=None):
    """Generate points on a regular polygon."""
    if center is None:
        center = np.array([0.0, 0.0])
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return center + radius * np.column_stack((np.cos(angles), np.sin(angles)))


def _fermat_spiral(n):
    """Generate points using Fermat's spiral (Vogel's model)."""
    i = np.arange(n)
    r = np.sqrt(i / n)
    theta = i * 2.399963229728653  # Golden angle in radians
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    points = np.column_stack((x, y))
    # Center the points
    points -= points.mean(axis=0)
    return points


def _optimize_from_start(starting_points, max_time=100):
    """Optimize from a given starting point using Powell's method."""
    start_time = time.time()
    pts = starting_points.copy()
    current_score = _score(pts)
    
    def objective(params):
        pts = params.reshape(-1, 2)
        return -_score(pts)
    
    try:
        result = minimize(objective, pts.flatten(), method='Powell', 
                         options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
        if time.time() - start_time > max_time:
            return pts, current_score
        optimized = result.x.reshape(-1, 2)
        opt_score = _score(optimized)
        if opt_score > current_score:
            return optimized, opt_score
    except Exception:
        pass
    
    return pts, current_score


def construct():
    """Construct 13 points to maximize min triangle area / convex hull area."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_points = None
    best_score = 0.0
    
    # Generate diverse starting points
    starts = []
    
    # 1. Regular 13-gon
    starts.append(_regular_polygon(13))
    
    # 2. Fermat spiral
    starts.append(_fermat_spiral(13))
    
    # 3. Grid-like pattern
    grid_pts = []
    for i in range(4):
        for j in range(4):
            if len(grid_pts) < 13:
                grid_pts.append([i * 0.3 - 0.45, j * 0.3 - 0.45])
    starts.append(np.array(grid_pts))
    
    # 4. Random starts with fixed seed
    rng = np.random.RandomState(42)
    for _ in range(5):
        pts = rng.rand(13, 2)
        pts -= pts.mean(axis=0)
        starts.append(pts)
    
    # 5. Ring structure: 1 center + 12 on circle
    ring_pts = np.zeros((13, 2))
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    ring_pts[1:, 0] = np.cos(angles)
    ring_pts[1:, 1] = np.sin(angles)
    starts.append(ring_pts)
    
    # Optimize from each start
    for i, start_pts in enumerate(starts):
        if time.time() > deadline:
            break
        remaining_time = deadline - time.time()
        if remaining_time < 5:
            break
        max_iter_time = min(remaining_time - 2, 30)  # Leave buffer
        if max_iter_time < 2:
            break
        
        opt_pts, opt_score = _optimize_from_start(start_pts, max_time=max_iter_time)
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_pts.copy()
    
    # If no optimization succeeded, use the best unoptimized start
    if best_points is None:
        for start_pts in starts:
            s = _score(start_pts)
            if s > best_score:
                best_score = s
                best_points = start_pts.copy()
    
    if best_points is None:
        # Fallback: regular polygon
        best_points = _regular_polygon(13)
    
    return best_points

# EVOLVE-BLOCK-END
