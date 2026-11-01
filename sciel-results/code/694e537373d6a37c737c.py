"""Heilbronn convex regions n=14: maximize min triangle area / convex hull area."""
N = 14
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
import itertools

def _min_triangle_area(pts):
    """Compute the minimum area of all triples of points."""
    n = len(pts)
    min_area = float('inf')
    # Use vectorized computation for speed
    # For each pair (i, j), compute cross product with all other points k
    # Area of triangle (i, j, k) = 0.5 * |cross(pts[j]-pts[i], pts[k]-pts[i])|
    # We can vectorize over k for each (i,j) pair
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            v1 = pts[j] - pts[i]
            # Cross product of v1 with (pts[k] - pts[i]) for all k
            # cross(a, b) = a[0]*b[1] - a[1]*b[0]
            diffs = pts - pts[i]
            cross = v1[0] * diffs[:, 1] - v1[1] * diffs[:, 0]
            # We only want k > j to avoid duplicates, but for min we can check all k != i, j
            # Actually, let's just compute abs of all and take min, excluding i and j
            areas = np.abs(cross) * 0.5
            # Exclude i and j (they give 0 area)
            areas[i] = float('inf')
            areas[j] = float('inf')
            current_min = np.min(areas)
            if current_min < min_area:
                min_area = current_min
    return min_area

def _score(pts):
    """Score = min triangle area / convex hull area."""
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume  # In 2D, volume is area
        if hull_area < 1e-12:
            return 0.0
        min_area = _min_triangle_area(pts)
        return min_area / hull_area
    except Exception:
        return 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Remove translation invariance: fix p0 at origin
    # Remove rotation invariance: fix p1 on positive x-axis
    # So we optimize 2*(N-2) = 24 variables
    
    def get_points(x):
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        for i in range(N - 2):
            pts[i + 2, 0] = x[2 * i]
            pts[i + 2, 1] = x[2 * i + 1]
        return pts
    
    def objective(x):
        pts = get_points(x)
        # Check for duplicate points
        min_dist = float('inf')
        for i in range(N - 1):
            for j in range(i + 1, N):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < min_dist:
                    min_dist = d
        if min_dist < 1e-10:
            return -1e10
        s = _score(pts)
        return -s  # Minimize negative score
    
    # Initial bounds: points should be in a reasonable range
    # Since we fixed p0=(0,0) and p1=(1,0), other points should be within [-5, 5]
    bounds = [(-5, 5)] * (2 * (N - 2))
    
    # Use differential evolution for global search
    best_result = None
    best_score = 0.0
    
    # Try a few different initial strategies
    # Strategy 1: Regular polygon-like initial guess
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    init_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    # Normalize so that p0 is at origin and p1 is at (1,0)
    # This is tricky; let's just use DE with reasonable bounds
    
    # Run differential evolution
    try:
        de_result = differential_evolution(
            objective,
            bounds=bounds,
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-8,
            polish=True,
            callback=None
        )
        if de_result.success or de_result.fun < 0:
            best_x = de_result.x
            best_score = -de_result.fun
            best_result = get_points(best_x)
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        if best_result is not None:
            return best_result
        # Fallback: regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        return np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Refine with local optimization if time permits
    if best_result is not None:
        try:
            refined = minimize(
                objective,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if refined.fun < de_result.fun:
                best_x = refined.x
                best_score = -refined.fun
                best_result = get_points(best_x)
        except Exception:
            pass
    
    # If no result from DE, try a simple regular polygon
    if best_result is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_result = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Check time
    if time.time() > deadline - 5:
        return best_result
    
    # Try a second DE run with different seed if time permits
    if time.time() < deadline - 40:
        try:
            de_result2 = differential_evolution(
                objective,
                bounds=bounds,
                seed=123,
                maxiter=80,
                popsize=12,
                tol=1e-8,
                polish=True
            )
            if de_result2.fun < de_result.fun:
                best_x = de_result2.x
                best_score = -de_result2.fun
                best_result = get_points(best_x)
        except Exception:
            pass
    
    return best_result
