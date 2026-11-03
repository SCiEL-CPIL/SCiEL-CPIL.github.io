"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from itertools import combinations

DEADLINE = 140.0  # Safety margin before 150s kill


def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _all_triangle_areas(pts):
    """Compute areas of all C(n,3) triangles."""
    n = len(pts)
    areas = np.empty((n*(n-1)*(n-2))//6)
    idx = 0
    for i in range(n-2):
        for j in range(i+1, n-1):
            for k in range(j+1, n):
                ax, ay = pts[i]
                bx, by = pts[j]
                cx, cy = pts[k]
                area = 0.5 * abs((bx-ax)*(cy-ay) - (cx-ax)*(by-ay))
                areas[idx] = area
                idx += 1
    return areas


def _score(pts):
    """Compute min triangle area / hull area."""
    hull_a = _hull_area(pts)
    if hull_a <= 0:
        return 0.0
    areas = _all_triangle_areas(pts)
    return np.min(areas) / hull_a


def _neg_score(pts_flat):
    """Negative score for minimization."""
    pts = pts_flat.reshape(-1, 2)
    s = _score(pts)
    return -s


def _initial_regular_polygon(n):
    """Points on a regular n-gon inscribed in unit circle, scaled to unit area."""
    angles = 2 * np.pi * np.arange(n) / n
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    # Scale to unit area
    hull_a = _hull_area(pts)
    if hull_a > 0:
        pts /= np.sqrt(hull_a)
    return pts


def _initial_mixed(n):
    """Mix of boundary and interior points for better minimum triangle area."""
    # Use 10 points on outer ring, 4 interior
    outer_n = 10
    inner_n = n - outer_n
    angles = 2 * np.pi * np.arange(outer_n) / outer_n
    outer = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Interior points: small regular polygon offset
    inner_angles = 2 * np.pi * np.arange(inner_n) / inner_n + np.pi / inner_n
    inner = 0.4 * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    
    pts = np.vstack([outer, inner])
    hull_a = _hull_area(pts)
    if hull_a > 0:
        pts /= np.sqrt(hull_a)
    return pts


def construct():
    start_time = time.time()
    
    # Try multiple initial configurations
    inits = [
        _initial_regular_polygon(N),
        _initial_mixed(N),
    ]
    
    # Also try a few random perturbations of the regular polygon
    rng = np.random.RandomState(42)
    for _ in range(3):
        base = _initial_regular_polygon(N)
        noise = rng.normal(0, 0.05, size=base.shape)
        perturbed = base + noise
        hull_a = _hull_area(perturbed)
        if hull_a > 0:
            perturbed /= np.sqrt(hull_a)
            inits.append(perturbed)
    
    best_pts = None
    best_score = 0.0
    
    for init in inits:
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Optimize from this initial point
        x0 = init.flatten()
        try:
            res = minimize(_neg_score, x0, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8,
                                  'adaptive': True})
            if res.x is not None:
                cand = res.x.reshape(-1, 2)
                s = _score(cand)
                if s > best_score:
                    best_score = s
                    best_pts = cand
        except Exception:
            pass
    
    # Final polish with tighter tolerance if time permits
    if best_pts is not None and time.time() - start_time < DEADLINE - 10:
        try:
            x0 = best_pts.flatten()
            res = minimize(_neg_score, x0, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10,
                                  'adaptive': True})
            if res.x is not None:
                cand = res.x.reshape(-1, 2)
                s = _score(cand)
                if s > best_score:
                    best_score = s
                    best_pts = cand
        except Exception:
            pass
    
    if best_pts is None:
        # Fallback: regular polygon
        best_pts = _initial_regular_polygon(N)
    
    return best_pts
# EVOLVE-BLOCK-END
