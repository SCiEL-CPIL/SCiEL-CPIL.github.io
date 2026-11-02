import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 14

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    # Pre-compute all triangle index combinations
    tri_indices = list(combinations(range(N), 3))
    tri_indices = np.array(tri_indices)  # shape (364, 3)
    
    # Initial guess: Regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
    
    def compute_score(pts):
        """Compute min triangle area / convex hull area"""
        # Convex Hull Area
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume  # For 2D, .volume is area
            if hull_area < 1e-10:
                return 0.0
        except Exception:
            return 0.0
        
        # Vectorized min triangle area computation
        # For each triangle (i,j,k), area = 0.5 * |cross(pts[j]-pts[i], pts[k]-pts[i])|
        pi = pts[tri_indices[:, 0]]  # shape (364, 2)
        pj = pts[tri_indices[:, 1]]  # shape (364, 2)
        pk = pts[tri_indices[:, 2]]  # shape (364, 2)
        
        v1 = pj - pi  # shape (364, 2)
        v2 = pk - pi  # shape (364, 2)
        
        # Cross product: v1[0]*v2[1] - v1[1]*v2[0]
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        min_area = np.min(areas)
        score = min_area / hull_area
        return score
    
    def neg_score(flat_pts):
        pts = flat_pts.reshape(N, 2)
        # Check for degeneracy
        if np.allclose(pts, pts[0]):
            return 0.0
        return -compute_score(pts)
    
    # Optimization with multiple restarts
    best_score_val = 0
    best_result = x0.copy()
    
    # Try multiple initial points and methods
    initial_points = [
        x0,  # Regular 14-gon
        x0 + np.random.normal(0, 0.05, x0.shape),  # Perturbed regular
        x0 + np.random.normal(0, 0.1, x0.shape),   # More perturbed
    ]
    
    methods = ['Powell', 'Nelder-Mead']
    
    for x_start in initial_points:
        if time.time() > deadline - 20:
            break
        for method in methods:
            if time.time() > deadline - 10:
                break
            try:
                if method == 'Powell':
                    res = minimize(neg_score, x_start, method='Powell',
                                   options={'maxiter': 500, 'ftol': 1e-10, 'disp': False})
                else:
                    res = minimize(neg_score, x_start, method='Nelder-Mead',
                                   options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': False})
                
                score_val = -res.fun
                if score_val > best_score_val:
                    best_score_val = score_val
                    best_result = res.x.copy()
            except Exception:
                pass
    
    # Final polish: use the best result as starting point with tighter tolerance
    if time.time() < deadline - 5 and best_result is not None:
        try:
            res_polish = minimize(neg_score, best_result, method='Powell',
                                  options={'maxiter': 200, 'ftol': 1e-14, 'disp': False})
            score_val = -res_polish.fun
            if score_val > best_score_val:
                best_score_val = score_val
                best_result = res_polish.x.copy()
        except Exception:
            pass
    
    return best_result.reshape(N, 2)
