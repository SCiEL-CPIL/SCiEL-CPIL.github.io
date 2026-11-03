"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

_start_time = time.time()
_TIME_LIMIT = 140.0  # safety margin before 150s kill

def _time_left():
    return _TIME_LIMIT - (time.time() - _start_time)

def _score(points):
    """Compute min triangle area / convex hull area."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    n = points.shape[0]
    if n < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
    if hull_area <= 1e-12:
        return 0.0
    # Compute all triangle areas
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                if area < min_area:
                    min_area = area
    return min_area / hull_area

def _neg_score(points):
    """Negative score for minimization."""
    return -_score(points)

def _regular_ngon(n, radius=1.0, center=(0.0, 0.0)):
    """Generate vertices of a regular n-gon."""
    angles = 2.0 * np.pi * np.arange(n) / n
    xs = center[0] + radius * np.cos(angles)
    ys = center[1] + radius * np.sin(angles)
    return np.column_stack([xs, ys])

def construct():
    global _start_time
    _start_time = time.time()
    
    # Start with regular 14-gon
    best_points = _regular_ngon(N, radius=1.0)
    best_score = _score(best_points)
    
    # Try optimization to improve
    if _time_left() > 30.0:
        # Use Nelder-Mead on the 2N coordinates
        x0 = best_points.flatten()
        try:
            result = minimize(
                _neg_score,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if result.fun < -best_score:
                candidate = result.x.reshape(N, 2)
                s = _score(candidate)
                if s > best_score:
                    best_points = candidate
                    best_score = s
        except Exception:
            pass
    
    # If time remains, try a few random perturbations + local optimization
    if _time_left() > 20.0:
        rng = np.random.RandomState(42)
        for trial in range(20):
            if _time_left() < 15.0:
                break
            # Perturb the best points slightly
            perturbed = best_points + rng.normal(0, 0.05, size=(N, 2))
            x0 = perturbed.flatten()
            try:
                result = minimize(
                    _neg_score,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                candidate = result.x.reshape(N, 2)
                s = _score(candidate)
                if s > best_score:
                    best_points = candidate
                    best_score = s
            except Exception:
                pass
    
    return best_points
# EVOLVE-BLOCK-END
