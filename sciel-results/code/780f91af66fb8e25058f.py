"""Heilbronn convex regions: place 13 points to maximize min triangle area / hull area."""
import numpy as np
from itertools import combinations
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13

def _min_triangle_area(pts):
    """Compute minimum triangle area over all triples."""
    n = len(pts)
    min_a = np.inf
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                a = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                if a < min_a:
                    min_a = a
    return min_a

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: shoelace on convex hull via sorting
        # Use scipy fallback
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    ha = _hull_area(pts)
    if ha <= 1e-12:
        return 0.0
    ma = _min_triangle_area(pts)
    return ma / ha

def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
    """Generate n points on a regular polygon."""
    angles = 2 * np.pi * np.arange(n) / n
    xs = center[0] + radius * np.cos(angles)
    ys = center[1] + radius * np.sin(angles)
    return np.column_stack([xs, ys])

def construct():
    deadline = time.time() + 100.0  # Safety margin before 120s limit
    
    # Start with regular 13-gon
    best_pts = _regular_polygon(N, radius=1.0)
    best_score = _score(best_pts)
    
    # Objective: minimize negative score
    def neg_score(x):
        pts = x.reshape(N, 2)
        s = _score(pts)
        return -s
    
    # Try multiple initializations and optimizations
    inits = []
    
    # Regular polygon
    inits.append(_regular_polygon(N, radius=1.0).flatten())
    
    # Regular polygon with slight perturbation
    rng = np.random.RandomState(42)
    for trial in range(5):
        base = _regular_polygon(N, radius=1.0)
        noise = rng.randn(N, 2) * 0.1
        inits.append((base + noise).flatten())
    
    # Try with some interior points: 10 on hull, 3 interior
    for trial in range(3):
        hull_pts = _regular_polygon(10, radius=1.0)
        # Place 3 interior points
        interior = rng.rand(3, 2) * 0.6 + 0.2  # In [0.2, 0.8] roughly
        # Center around origin
        interior = interior - 0.5
        all_pts = np.vstack([hull_pts, interior])
        # Shuffle
        idx = rng.permutation(N)
        inits.append(all_pts[idx].flatten())
    
    # Optimize each initialization
    for init in inits:
        if time.time() > deadline:
            break
        
        try:
            result = minimize(neg_score, init, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8,
                                     'adaptive': True})
            if result.fun < -best_score:
                best_score = -result.fun
                best_pts = result.x.reshape(N, 2)
        except Exception:
            pass
    
    # Additional local refinement on best
    if time.time() < deadline:
        try:
            result = minimize(neg_score, best_pts.flatten(), method='Nelder-Mead',
                             options={'maxiter': 10000, 'xatol': 1e-7, 'fatol': 1e-9,
                                     'adaptive': True})
            if result.fun < -best_score:
                best_score = -result.fun
                best_pts = result.x.reshape(N, 2)
        except Exception:
            pass
    
    # Normalize: scale so that convex hull area = 1
    # The score is invariant to scaling, but let's ensure valid output
    ha = _hull_area(best_pts)
    if ha > 1e-12:
        scale = 1.0 / np.sqrt(ha)
        best_pts = best_pts * scale
    
    return best_pts
