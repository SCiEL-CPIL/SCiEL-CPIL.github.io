"""Heilbronn convex regions n=14: maximize minimum triangle area / hull area."""
import numpy as np
import time
from itertools import combinations
from scipy.optimize import differential_evolution, minimize

N = 14
TIME_LIMIT = 110.0  # safety margin before 120s kill


def _min_triangle_area(pts):
    """Compute the minimum triangle area among all triples of points."""
    n = len(pts)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(pts):
    """Compute the area of the convex hull of the points."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(pts)
        return hull.volume  # in 2D, volume is area
    except Exception:
        # Fallback: shoelace on sorted points (less robust)
        angles = np.arctan2(pts[:, 1] - pts[1, 1], pts[:, 0] - pts[1, 0])
        order = np.argsort(angles)
        ordered = pts[order]
        x, y = ordered[:, 0], ordered[:, 1]
        return 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-12:
        return 0.0
    min_tri = _min_triangle_area(pts)
    return min_tri / hull_area


def _regular_polygon(n):
    """Generate a regular n-gon centered at origin with unit circumradius."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _perturb_regular(pts, seed):
    """Create a perturbed regular polygon for diverse initial guesses."""
    rng = np.random.RandomState(seed)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.uniform(-0.1, 0.1, N)
    radii = 1.0 + rng.uniform(-0.15, 0.15, N)
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return pts


def construct():
    start_time = time.time()
    
    # Initial candidate: regular 14-gon
    best_pts = _regular_polygon(N)
    best_score = _score(best_pts)
    
    # Try several perturbed regular polygons as starting points
    for seed in range(5):
        if time.time() - start_time > TIME_LIMIT - 30:
            break
        cand = _perturb_regular(best_pts, seed)
        s = _score(cand)
        if s > best_score:
            best_score = s
            best_pts = cand
    
    # Local refinement using Nelder-Mead on the point coordinates
    # We optimize the negative score (since we maximize)
    def neg_score(pts_flat):
        pts = pts_flat.reshape(N, 2)
        return -_score(pts)
    
    # Use multiple restarts of local optimization
    for restart in range(8):
        if time.time() - start_time > TIME_LIMIT - 20:
            break
        
        if restart == 0:
            x0 = best_pts.ravel()
        else:
            x0 = _perturb_regular(best_pts, restart + 10).ravel()
        
        try:
            result = minimize(
                neg_score,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-12,
                    'fatol': 1e-14,
                    'adaptive': True
                }
            )
            if result.fun < -best_score:
                best_score = -result.fun
                best_pts = result.x.reshape(N, 2)
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() - start_time < TIME_LIMIT - 10:
        try:
            def neg_score_final(pts_flat):
                pts = pts_flat.reshape(N, 2)
                return -_score(pts)
            
            result = minimize(
                neg_score_final,
                best_pts.ravel(),
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-14,
                    'fatol': 1e-16,
                    'adaptive': True
                }
            )
            if result.fun < -best_score:
                best_score = -result.fun
                best_pts = result.x.reshape(N, 2)
        except Exception:
            pass
    
    # Normalize: center at origin and scale so hull area is 1
    # Actually, the score is invariant to translation and scaling,
    # but let's normalize for cleanliness
    cx, cy = np.mean(best_pts, axis=0)
    best_pts = best_pts - np.array([cx, cy])
    
    return best_pts
