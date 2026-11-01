import numpy as np
from scipy.optimize import minimize
import time

N = 14

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s out of 150s limit

    # Strategy: 2 concentric rings of 7 points each.
    # Outer ring: radius 1.0, angles 0, 2pi/7, ...
    # Inner ring: radius r, angles offset by pi/7.
    
    def generate_points(r, offset):
        angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        angles_inner = angles_outer + offset
        pts_outer = np.column_stack((np.cos(angles_outer), np.sin(angles_outer)))
        pts_inner = np.column_stack((r * np.cos(angles_inner), r * np.sin(angles_inner)))
        return np.vstack([pts_outer, pts_inner])

    def score(pts):
        # Calculate convex hull area
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume # In 2D, volume is area
        except:
            return 0.0
        
        # Calculate min triangle area
        min_area = float('inf')
        n = len(pts)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    p1 = pts[i]
                    p2 = pts[j]
                    p3 = pts[k]
                    # Area = 0.5 * |det(p2-p1, p3-p1)|
                    v1 = p2 - p1
                    v2 = p3 - p1
                    area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                    if area < min_area:
                        min_area = area
                        if min_area < 1e-10:
                            return 0.0
                            
        return min_area / hull_area

    def objective(params):
        r = params[0]
        offset = params[1]
        if r < 0.1 or r > 0.9:
            return -1.0
        pts = generate_points(r, offset)
        return -score(pts) # Minimize negative score

    # Initial guess: r=0.5, offset=pi/7
    x0 = [0.5, np.pi/7]
    
    # Try a few random starts and local optimizations
    best_score = 0.0
    best_params = None
    
    # Grid search for r
    for r in np.linspace(0.2, 0.8, 15):
        for offset in np.linspace(0, np.pi/7, 5):
            if time.time() > deadline:
                break
            pts = generate_points(r, offset)
            s = score(pts)
            if s > best_score:
                best_score = s
                best_params = [r, offset]
        
        if time.time() > deadline:
            break

    # Local optimization from best grid point
    if best_params is not None:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead', 
                           options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 200})
            if res.fun < 0:
                s_opt = -res.fun
                if s_opt > best_score:
                    best_score = s_opt
                    best_params = res.x
        except Exception:
            pass

    # Generate final points
    if best_params is not None:
        r, offset = best_params
        return generate_points(r, offset)
    else:
        # Fallback to regular 14-gon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        return np.column_stack((np.cos(angles), np.sin(angles)))
