import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
import itertools

N = 14

def _triangle_areas(pts):
    """Compute areas of all triangles formed by triples of points."""
    n = len(pts)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = pts[i]
                b = pts[j]
                c = pts[k]
                area = 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))
                areas.append(area)
    return np.array(areas)

def _score(pts):
    """Compute min triangle area divided by convex hull area."""
    try:
        areas = _triangle_areas(pts)
        min_area = np.min(areas)
        hull = ConvexHull(pts)
        hull_area = hull.volume  # In 2D, volume is area
        if hull_area <= 0:
            return 0.0
        return min_area / hull_area
    except Exception:
        return 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initialize with regular 14-gon on unit circle
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    best_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    best_score = _score(best_pts)
    
    def objective(x):
        """Negative score for minimization."""
        pts = x.reshape(-1, 2)
        s = _score(pts)
        return -s
    
    # Try local optimization from the regular polygon
    x0 = best_pts.flatten()
    
    # Use Nelder-Mead which handles nonsmooth objectives
    try:
        result = minimize(objective, x0, method='Nelder-Mead',
                         options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
        if time.time() < deadline:
            cand_pts = result.x.reshape(-1, 2)
            cand_score = _score(cand_pts)
            if cand_score > best_score:
                best_pts = cand_pts
                best_score = cand_score
    except Exception:
        pass
    
    # Try a few perturbations if time remains
    rng = np.random.RandomState(42)
    for trial in range(20):
        if time.time() >= deadline:
            break
        
        # Small random perturbation of the best solution
        perturbation_scale = 0.01 * (1.0 - trial / 20.0)
        perturbed = best_pts + rng.normal(0, perturbation_scale, size=(N, 2))
        
        try:
            x0 = perturbed.flatten()
            result = minimize(objective, x0, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            cand_pts = result.x.reshape(-1, 2)
            cand_score = _score(cand_pts)
            if cand_score > best_score:
                best_pts = cand_pts
                best_score = cand_score
        except Exception:
            continue
    
    return best_pts
