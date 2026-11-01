"""Optimized Heilbronn convex regions n=14 solver."""
import numpy as np
import time
import itertools
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

def _min_triangle_area(pts):
    """Compute the minimum triangle area among all triples of points."""
    n = len(pts)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross product|
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(pts):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, ConvexHull.volume gives area
    except Exception:
        # Fallback: use shoelace on sorted points
        # This shouldn't happen if points are in general position
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    min_ta = _min_triangle_area(pts)
    ch_area = _convex_hull_area(pts)
    if ch_area <= 1e-12:
        return 0.0
    return min_ta / ch_area

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Start with regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    regular_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Fix p0 and p1 to remove translation invariance
    # p0 = (0, 0), p1 = (1, 0)
    # The regular polygon has p0=(1,0), p1=(cos(2pi/14), sin(2pi/14))
    # We'll work in a coordinate system where p0=(0,0) and p1=(1,0)
    
    # For optimization, fix first two points
    # Initial: use regular polygon, then translate/rotate so p0=(0,0), p1=(1,0)
    p0 = regular_pts[0].copy()
    p1 = regular_pts[1].copy()
    
    # Translate so p0 is at origin
    shifted = regular_pts - p0
    # Now p0 is at (0,0), p1 is at some position
    d = shifted[1]
    dist = np.linalg.norm(d)
    # Rotate so p1 is at (1, 0)
    theta = np.arctan2(d[1], d[0])
    rot_matrix = np.array([[np.cos(-theta), -np.sin(-theta)],
                           [np.sin(-theta), np.cos(-theta)]])
    rotated = shifted @ rot_matrix.T
    # Scale so p1 is at (1, 0)
    rotated = rotated / dist
    
    best_pts = rotated.copy()
    best_score = _score(best_pts)
    
    # Nelder-Mead optimization on the free points (p2 through p13)
    # Variables: 2*(N-2) = 24 coordinates
    
    def objective(x):
        """Negative score to minimize."""
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        for i in range(N - 2):
            pts[i + 2, 0] = x[2 * i]
            pts[i + 2, 1] = x[2 * i + 1]
        return -_score(pts)
    
    # Initial x from best_pts
    x0 = np.zeros(2 * (N - 2))
    for i in range(N - 2):
        x0[2 * i] = best_pts[i + 2, 0]
        x0[2 * i + 1] = best_pts[i + 2, 1]
    
    # Run Nelder-Mead
    try:
        result = minimize(objective, x0, method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
        
        if result.fun < -best_score:
            x_opt = result.x
            best_pts = np.zeros((N, 2))
            best_pts[0] = [0.0, 0.0]
            best_pts[1] = [1.0, 0.0]
            for i in range(N - 2):
                best_pts[i + 2, 0] = x_opt[2 * i]
                best_pts[i + 2, 1] = x_opt[2 * i + 1]
            best_score = -result.fun
    except Exception:
        pass
    
    # If we have time, try a second round with tighter tolerances
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        try:
            result2 = minimize(objective, x0, method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12,
                                       'initial_simplex': None})
            if result2.fun < -best_score:
                x_opt = result2.x
                best_pts = np.zeros((N, 2))
                best_pts[0] = [0.0, 0.0]
                best_pts[1] = [1.0, 0.0]
                for i in range(N - 2):
                    best_pts[i + 2, 0] = x_opt[2 * i]
                    best_pts[i + 2, 1] = x_opt[2 * i + 1]
                best_score = -result2.fun
        except Exception:
            pass
    
    # Check deadline
    if time.time() > deadline:
        pass  # Return best found so far
    
    return best_pts

# EVOLVE-BLOCK-END
