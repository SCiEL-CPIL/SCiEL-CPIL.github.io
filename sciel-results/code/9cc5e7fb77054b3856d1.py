"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from itertools import combinations

N = 14

# EVOLVE-BLOCK-START
def _triangle_areas(pts):
    """Compute areas of all triangles formed by triples of points."""
    n = len(pts)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                xi, yi = pts[i]
                xj, yj = pts[j]
                xk, yk = pts[k]
                area = 0.5 * abs((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi))
                areas.append(area)
    return np.array(areas)

def _score(pts):
    """Score = min triangle area / convex hull area."""
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume  # In 2D, .volume gives area
        if hull_area < 1e-12:
            return 0.0
        areas = _triangle_areas(pts)
        min_area = np.min(areas)
        return min_area / hull_area
    except Exception:
        return 0.0

def _objective(x):
    """Negative score for minimization."""
    pts = x.reshape(-1, 2)
    return -_score(pts)

def construct():
    deadline = time.time() + 130  # Safety margin before 150s kill
    
    # Start with regular polygon (good initial guess)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    best_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    best_score = _score(best_pts)
    
    # Try to improve with local optimization
    # We fix p0 at (0,0) and p1 at (1,0) to remove translational invariance
    # This reduces variables from 2*N to 2*(N-2)
    
    # Normalize initial points: translate so p0 is at origin
    # Then rotate so p1 is at (1,0)
    def _normalize_for_opt(pts):
        """Remove translational and rotational invariance.
        Fix p0 at (0,0), p1 at (1,0)."""
        pts = pts.copy()
        # Translate
        pts = pts - pts[0]
        # Rotate so that p1 is along x-axis
        p1 = pts[1]
        angle = np.arctan2(p1[1], p1[0])
        c, s = np.cos(-angle), np.sin(-angle)
        R = np.array([[c, -s], [s, c]])
        pts = pts @ R.T
        # Scale so that p1 is at (1,0)
        dist = np.linalg.norm(p1)
        if dist > 1e-12:
            pts = pts / dist
        return pts
    
    def _from_opt_vars(x):
        """Reconstruct full point set from optimization variables."""
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        for i in range(N - 2):
            pts[i + 2, 0] = x[2 * i]
            pts[i + 2, 1] = x[2 * i + 1]
        return pts
    
    def _objective_norm(x):
        pts = _from_opt_vars(x)
        return -_score(pts)
    
    # Try multiple random restarts with local optimization
    np.random.seed(N)
    
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Generate random initial configuration
        if restart == 0:
            # Use the regular polygon as first start
            init_pts = best_pts.copy()
        else:
            # Random points in a disk
            r = np.sqrt(np.random.rand(N))
            theta = 2 * np.pi * np.random.rand(N)
            init_pts = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        
        # Normalize for optimization
        norm_pts = _normalize_for_opt(init_pts)
        x0 = norm_pts[2:].flatten()
        
        # Bounds: points should be in a reasonable range
        bounds = [(-2, 2)] * (2 * (N - 2))
        
        # Use L-BFGS-B with multiple iterations
        try:
            result = minimize(
                _objective_norm,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.fun < -best_score:
                new_pts = _from_opt_vars(result.x)
                new_score = -result.fun
                if new_score > best_score:
                    best_pts = new_pts
                    best_score = new_score
        except Exception:
            pass
    
    return best_pts
# EVOLVE-BLOCK-END
