"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 13

# EVOLVE-BLOCK-START

def _triangle_areas(points):
    """Compute areas of all triangles formed by triples of points."""
    n = len(points)
    areas = np.zeros((n, n, n))
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
                areas[i, j, k] = area
    return areas

def _score(points):
    """
    Calculate the minimum triangle area divided by the convex hull area.
    Returns 0 if points are collinear or hull area is 0.
    """
    if len(points) < 3:
        return 0.0
    
    # Compute convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume  # In 2D, volume is area
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
    # Compute minimum triangle area
    areas = _triangle_areas(points)
    # Only consider upper triangle (i < j < k) to avoid duplicates and zeros
    min_area = np.inf
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            for k in range(j+1, len(points)):
                if areas[i, j, k] < min_area:
                    min_area = areas[i, j, k]
    
    if min_area <= 1e-12:
        return 0.0
    
    return min_area / hull_area

def _normalize(points):
    """
    Normalize points so that their convex hull has unit area.
    If hull area is zero or very small, return points as-is (or centered).
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-12:
            # Center the points to avoid numerical issues
            center = np.mean(points, axis=0)
            return points - center
        scale = 1.0 / np.sqrt(hull_area)
        center = np.mean(points, axis=0)
        return (points - center) * scale
    except Exception:
        center = np.mean(points, axis=0)
        return points - center

def _objective(x, deadline):
    """
    Objective function for minimization: minimize negative score.
    x is a flat array of 2*N coordinates.
    """
    if time.time() > deadline:
        return 1e10
    
    points = x.reshape(N, 2)
    # Normalize to unit hull area
    points_norm = _normalize(points)
    score = _score(points_norm)
    if score <= 0:
        return 1e10
    return -score

def _random_initial(rng, n=N):
    """Generate random points in the unit square."""
    return np.array([rng.uniform(-1, 1, 2) for _ in range(n)])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = 0.0
    
    # Strategy: Local search on the normalized point set
    # We optimize the raw points, but the score is computed on normalized points.
    # Alternatively, we can optimize points directly in a normalized space (e.g., unit circle or square)
    # and then normalize. However, normalizing changes the geometry slightly if not done carefully.
    # A better approach: Optimize points such that their hull area is 1.
    # We can parameterize points in a way that preserves hull area or just normalize after each step.
    
    # Let's try optimizing the points directly, but using a penalty or constraint for hull area?
    # No, the task says "normalized to unit area". The evaluator normalizes by the point set convex hull area.
    # So the score is invariant to scaling. We can just optimize in any scale and the score will be the same.
    # But we need to be careful: if we scale points, the minimum triangle area scales by the same factor as the hull area.
    # So the ratio is invariant to scaling.
    # Therefore, we can fix the hull area to 1 by normalizing, or just optimize the ratio directly.
    # Optimizing the ratio directly is simpler: maximize min_triangle_area / hull_area.
    
    # Let's use a local search approach.
    # Start with a good initial guess: points on a circle? Or a grid?
    # A regular 13-gon on a circle is a good starting point.
    
    # Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points_circle = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Normalize
    points_circle = _normalize(points_circle)
    score_circle = _score(points_circle)
    
    if score_circle > best_score:
        best_score = score_circle
        best_points = points_circle.copy()
    
    # Local search using Nelder-Mead or similar
    # Since the score is invariant to scaling, we can add a constraint or just let the optimizer find a good configuration.
    # But Nelder-Mead might drift in scale. We can normalize inside the objective.
    
    # Let's try multiple restarts with different initial configurations.
    
    initial_configs = [
        points_circle,
        _random_initial(rng),
        _random_initial(rng),
        # Maybe a grid-like structure
        np.array([[i % 4 - 1.5, (i // 4) - 1.5] for i in range(12)] + [[0, 0]]),
    ]
    
    for init_pts in initial_configs:
        if time.time() > deadline:
            break
        
        init_pts = _normalize(init_pts)
        init_flat = init_pts.flatten()
        
        # Use Nelder-Mead to maximize score (minimize -score)
        try:
            result = minimize(
                _objective,
                init_flat,
                args=(deadline,),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
            )
        except Exception:
            continue
        
        if time.time() > deadline:
            break
        
        if result.fun < -best_score:  # Since we minimize -score
            new_score = -result.fun
            new_points = _normalize(result.x.reshape(N, 2))
            # Verify score
            verified_score = _score(new_points)
            if verified_score > best_score:
                best_score = verified_score
                best_points = new_points.copy()
    
    # If we have time, do a more refined local search on the best points found
    if best_points is not None and time.time() < deadline - 10:
        # Try a finer local search
        init_flat = best_points.flatten()
        try:
            result = minimize(
                _objective,
                init_flat,
                args=(deadline,),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
        except Exception:
            pass
        
        if result.fun < -best_score:
            new_score = -result.fun
            new_points = _normalize(result.x.reshape(N, 2))
            verified_score = _score(new_points)
            if verified_score > best_score:
                best_score = verified_score
                best_points = new_points.copy()
    
    # Return the best points found
    if best_points is None:
        # Fallback to random points
        return _random_initial(rng)
    
    return best_points

# EVOLVE-BLOCK-END
