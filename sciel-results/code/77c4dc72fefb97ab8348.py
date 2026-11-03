"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools

def _halton_sequence(n, dim, base=None):
    """Generate n points in [0,1)^dim using Halton sequence."""
    if base is None:
        # Use first n primes as bases
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
        base = primes[:dim]
    
    points = []
    for i in range(1, n + 1):
        point = []
        for d, b in enumerate(base):
            f = 1.0
            r = 0.0
            i_val = i
            while i_val > 0:
                f /= b
                r += f * (i_val % b)
                i_val //= b
            point.append(r)
        points.append(point)
    return np.array(points)

def _min_triangle_area(pts):
    """Compute minimum triangle area for n points."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax, ay = pts[i]
                bx, by = pts[j]
                cx, cy = pts[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume = area
    except Exception:
        return 1.0

def _score(pts):
    """Score = min triangle area / hull area."""
    hull_a = _hull_area(pts)
    if hull_a <= 1e-15:
        return 0.0
    min_tri = _min_triangle_area(pts)
    return min_tri / hull_a

def _objective(x):
    """Objective to minimize: negative of score."""
    pts = x.reshape((-1, 2))
    return -_score(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Start with Halton sequence points
    best_pts = _halton_sequence(N, 2)
    best_score = _score(best_pts)
    
    # Try multiple Halton variants with different offsets
    for offset in range(1, 10):
        if time.time() > deadline:
            break
        pts = _halton_sequence(N, 2) + offset * 0.01
        pts = (pts - pts.min(axis=0)) / (pts.max(axis=0) - pts.min(axis=0) + 1e-15)
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
    
    # Gradient-based refinement using Nelder-Mead
    x0 = best_pts.flatten()
    for attempt in range(3):
        if time.time() > deadline:
            break
        
        # Add small random perturbation for diversity
        if attempt > 0:
            x0 = best_pts.flatten() + np.random.randn(N * 2) * 0.05
            x0 = np.clip(x0, 0.01, 0.99)
        
        try:
            result = minimize(_objective, x0, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12,
                                      'adaptive': True})
            if result.fun < -best_score:
                best_score = -result.fun
                best_pts = result.x.reshape((-1, 2))
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() < deadline - 5.0:
        try:
            result = minimize(_objective, best_pts.flatten(), method='Nelder-Mead',
                             options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14,
                                      'adaptive': True})
            if result.fun < -best_score:
                best_score = -result.fun
                best_pts = result.x.reshape((-1, 2))
        except Exception:
            pass
    
    return best_pts
# EVOLVE-BLOCK-END
