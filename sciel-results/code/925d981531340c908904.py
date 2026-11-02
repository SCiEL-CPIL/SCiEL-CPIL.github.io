"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

DEADLINE = 150.0
START_TIME = time.time()
SAFETY_MARGIN = 5.0


def _min_triangle_area(pts):
    """Compute minimum triangle area among all triples."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0


def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 0.0
    min_ta = _min_triangle_area(pts)
    return min_ta / hull_area


def _neg_score(pts):
    """Negative score for minimization."""
    return -_score(pts)


def _regular_polygon(n, radius=1.0):
    """Regular n-gon centered at origin."""
    angles = 2 * np.pi * np.arange(n) / n
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])


def construct():
    """Construct 13 points maximizing min triangle area / convex hull area."""
    # Start with regular 13-gon
    pts0 = _regular_polygon(N)
    
    # Normalize to unit area
    hull_area0 = _convex_hull_area(pts0)
    if hull_area0 > 0:
        scale = 1.0 / np.sqrt(hull_area0)
        pts0 = pts0 * scale
    
    best_pts = pts0.copy()
    best_score = _score(pts0)
    
    # Try differential evolution with bounded search
    # Search in a box around the regular polygon
    # Flatten points to 2D array
    def _flatten(pts):
        return pts.flatten()
    
    def _unflatten(x):
        return x.reshape(N, 2)
    
    # Bounds: search in a reasonable box
    # Regular polygon has radius ~1/sqrt(area), let's search in [-2, 2]^2
    bounds = [(-2, 2)] * (2 * N)
    
    # Use differential evolution with limited population and iterations
    # Check time periodically
    elapsed = time.time() - START_TIME
    remaining = DEADLINE - SAFETY_MARGIN - elapsed
    
    if remaining > 10:
        # Run differential evolution
        try:
            result = differential_evolution(
                _neg_score,
                bounds=bounds,
                args=(),
                seed=42,
                maxiter=50,
                popsize=10,
                tol=1e-6,
                polish=True,
                init='latinhypercube',
                callback=None
            )
            if result.success and result.fun < 0:
                candidate = _unflatten(result.x)
                cand_score = -result.fun
                if cand_score > best_score:
                    best_pts = candidate
                    best_score = cand_score
        except Exception:
            pass
    
    # Local refinement with Nelder-Mead
    elapsed = time.time() - START_TIME
    remaining = DEADLINE - SAFETY_MARGIN - elapsed
    
    if remaining > 5:
        try:
            result = minimize(
                _neg_score,
                _flatten(best_pts),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if result.fun < 0:
                candidate = _unflatten(result.x)
                cand_score = -result.fun
                if cand_score > best_score:
                    best_pts = candidate
                    best_score = cand_score
        except Exception:
            pass
    
    # Return the best points found
    return best_pts
# EVOLVE-BLOCK-END
