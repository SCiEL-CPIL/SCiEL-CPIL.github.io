import numpy as np
from scipy.optimize import minimize
import itertools

N = 11
SQRT3 = np.sqrt(3)

def construct():
    # Vertices of the equilateral triangle
    # A = (0, 0), B = (1, 0), C = (0.5, sqrt(3)/2)
    
    # Pre-compute indices for all triangles
    indices = list(itertools.combinations(range(N), 3))
    idx_i = np.array([t[0] for t in indices])
    idx_j = np.array([t[1] for t in indices])
    idx_k = np.array([t[2] for t in indices])
    
    def project_fast(points):
        """
        Fast heuristic projection to keep points inside the triangle.
        """
        x = points[:, 0].copy()
        y = points[:, 1].copy()
        
        # Clip x to [0, 1]
        x = np.clip(x, 0, 1)
        
        # y >= 0
        y = np.maximum(y, 0)
        
        # y <= sqrt(3) x
        y = np.minimum(y, SQRT3 * x)
        
        # y <= sqrt(3) (1 - x)
        y = np.minimum(y, SQRT3 * (1 - x))
        
        return np.column_stack((x, y))

    def min_area(points):
        """
        Calculate the minimum area of triangles formed by the points.
        """
        pi = points[idx_i]
        pj = points[idx_j]
        pk = points[idx_k]
        
        # Cross product (2D)
        cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
        areas = 0.5 * np.abs(cross)
        return np.min(areas)

    def objective(flat_pts):
        pts = flat_pts.reshape(N, 2)
        pts_proj = project_fast(pts)
        return -min_area(pts_proj)

    # Multi-start Nelder-Mead
    best_points = None
    best_score = -float('inf')
    
    np.random.seed(42)
    
    # Generate multiple random starting points
    n_starts = 100 # Increased starts for better coverage
    
    for start in range(n_starts):
        # Random points inside the triangle
        starts = []
        for _ in range(N):
            u = np.random.rand()
            v = np.random.rand()
            if u + v > 1:
                u = 1 - u
                v = 1 - v
            x = u + 0.5 * v
            y = (SQRT3 / 2) * v
            starts.append([x, y])
        
        start_flat = np.array(starts).flatten()
        
        try:
            res = minimize(objective, start_flat, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True})
            score = -res.fun
            if score > best_score:
                best_score = score
                best_points = res.x.reshape(N, 2)
        except Exception:
            pass

    if best_points is None:
        # Fallback
        best_points = np.random.rand(N, 2)
        best_points = project_fast(best_points)

    return best_points
