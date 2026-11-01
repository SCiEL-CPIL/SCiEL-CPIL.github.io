import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 14

def _triangle_areas(pts):
    """Compute all triangle areas for n points. Returns array of size C(n,3)."""
    n = len(pts)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross(pj - pi, pk - pi)|
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                a = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                areas.append(a)
    return np.array(areas)

def _score(pts):
    """Score = min triangle area / convex hull area."""
    try:
        areas = _triangle_areas(pts)
        min_area = np.min(areas)
        if min_area <= 0:
            return 0.0
        # Convex hull area
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume  # In 2D, volume is area
        except Exception:
            return 0.0
        if hull_area <= 0:
            return 0.0
        return min_area / hull_area
    except Exception:
        return 0.0

def _regular_polygon(n, radius=1.0):
    """Generate vertices of a regular n-gon inscribed in circle of given radius."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 130s to leave safety margin before 150s kill

    # Start with regular 14-gon
    pts = _regular_polygon(N, radius=1.0)
    best_pts = pts.copy()
    best_score = _score(best_pts)

    # Local optimization: minimize the negative of the score
    # We optimize all 28 coordinates
    def objective(x):
        p = x.reshape(N, 2)
        s = _score(p)
        return -s

    # Use Nelder-Mead for local refinement
    # Start from regular polygon
    x0 = best_pts.flatten()
    
    # Try a few rounds of optimization with different starting points
    attempts = [
        ("regular", x0.copy()),
    ]
    
    # Add some perturbed versions
    rng = np.random.default_rng(42)
    for trial in range(5):
        perturbed = best_pts + rng.normal(0, 0.05, size=(N, 2))
        attempts.append((f"perturb{trial}", perturbed.flatten()))
    
    # Also try slightly different radii
    for r in [0.9, 1.1]:
        p = _regular_polygon(N, radius=r)
        attempts.append((f"radius_{r}", p.flatten()))
    
    for name, x_start in attempts:
        if time.time() > deadline:
            break
        
        try:
            result = minimize(
                objective,
                x_start,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'maxfev': 30000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            if result.fun < -best_score:
                best_score = -result.fun
                best_pts = result.x.reshape(N, 2)
                
        except Exception:
            pass
    
    # If we have time, do more refinement from the best point
    if time.time() < deadline - 20:
        try:
            x_start = best_pts.flatten()
            result = minimize(
                objective,
                x_start,
                method='Nelder-Mead',
                options={
                    'maxiter': 10000,
                    'maxfev': 50000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            if result.fun < -best_score:
                best_score = -result.fun
                best_pts = result.x.reshape(N, 2)
        except Exception:
            pass

    # Check time before returning
    if time.time() > deadline:
        pass  # Just return what we have

    return best_pts

# EVOLVE-BLOCK-END
