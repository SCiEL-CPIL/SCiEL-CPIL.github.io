import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

N = 13

def compute_score(pts):
    # pts is (N, 2)
    # Check for degenerate hull
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume # In 2D, volume is area
        if hull_area <= 1e-10:
            return 1e10 # Penalize
    except Exception:
        return 1e10
        
    # Calculate min triangle area
    # Vectorized calculation of all triangle areas
    # i, j, k
    min_area = np.inf
    # Loop is fine for 286
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                p1 = pts[i]
                p2 = pts[j]
                p3 = pts[k]
                # Area = 0.5 * |cross(p2-p1, p3-p1)|
                v1 = p2 - p1
                v2 = p3 - p1
                area = 0.5 * np.abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_area:
                    min_area = area
    
    if min_area <= 1e-10:
        return 1e10
        
    return -min_area / hull_area

def construct():
    # Initial guess: Regular 13-gon
    theta0 = np.linspace(0, 2*np.pi, N, endpoint=False)
    x0 = np.cos(theta0)
    y0 = np.sin(theta0)
    pts0 = np.column_stack((x0, y0))
    
    # Flatten for optimizer
    x_flat0 = pts0.flatten()
    
    def objective(x_flat):
        pts = x_flat.reshape((N, 2))
        return compute_score(pts)
        
    # Try a few restarts with different initializations
    best_x = x_flat0
    best_score = objective(best_x)
    
    # Restart 1: Regular polygon
    res1 = minimize(objective, x_flat0, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
    if res1.fun < best_score:
        best_score = res1.fun
        best_x = res1.x
        
    # Restart 2: Jittered regular polygon
    np.random.seed(42)
    x_flat2 = x_flat0 + np.random.normal(0, 0.1, N*2)
    res2 = minimize(objective, x_flat2, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
    if res2.fun < best_score:
        best_score = res2.fun
        best_x = res2.x
        
    # Restart 3: Grid-like? Maybe a regular 12-gon + center?
    # Let's try a regular 12-gon plus one point in center
    theta12 = np.linspace(0, 2*np.pi, 12, endpoint=False)
    x12 = np.cos(theta12)
    y12 = np.sin(theta12)
    pts3 = np.column_stack((x12, y12))
    pts3 = np.vstack([pts3, [0, 0]])
    x_flat3 = pts3.flatten()
    res3 = minimize(objective, x_flat3, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
    if res3.fun < best_score:
        best_score = res3.fun
        best_x = res3.x

    best_pts = best_x.reshape((N, 2))
    return best_pts
