import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 14

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s limit
    
    # Initial guess for parameters
    # r: radius of inner heptagon (0 < r < 1)
    # theta: rotation of inner heptagon relative to outer
    # We assume outer heptagon has radius 1.
    
    def generate_points(r, theta):
        # Outer heptagon
        angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        outer_pts = np.column_stack((np.cos(angles_outer), np.sin(angles_outer)))
        
        # Inner heptagon
        angles_inner = angles_outer + theta
        inner_pts = r * np.column_stack((np.cos(angles_inner), np.sin(angles_inner)))
        
        # Interleave or just stack? 
        # The structure "7+7" usually implies two rings.
        # To maximize min triangle area, we want points well-distributed.
        # Stacking outer then inner is fine, the geometry is what matters.
        pts = np.vstack([outer_pts, inner_pts])
        return pts

    def compute_score(params):
        r, theta = params
        if r <= 0 or r >= 1:
            return -1.0 # Penalty for invalid r
        
        pts = generate_points(r, theta)
        
        # Check if inner points are strictly inside the outer hull
        # For a regular heptagon, if r < 1, they are inside.
        
        # Calculate convex hull area
        # The hull is the outer heptagon.
        # Area of regular n-gon with circumradius R:
        # A = 0.5 * n * R^2 * sin(2*pi/n)
        hull_area = 0.5 * 7 * 1.0 * 1.0 * np.sin(2 * np.pi / 7)
        
        # Calculate min triangle area
        min_area = np.inf
        n_pts = len(pts)
        # Vectorized calculation for speed
        # We need to check all C(14,3) = 364 triangles
        # We can do this efficiently using broadcasting or loops if not too slow.
        # 364 triangles is small.
        
        for i in range(n_pts - 2):
            for j in range(i + 1, n_pts - 1):
                for k in range(j + 1, n_pts):
                    p1 = pts[i]
                    p2 = pts[j]
                    p3 = pts[k]
                    # Area = 0.5 * | (p2-p1) x (p3-p1) |
                    v1 = p2 - p1
                    v2 = p3 - p1
                    area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                    if area < min_area:
                        min_area = area
                        if min_area == 0:
                            return -1.0
        
        if min_area == 0 or np.isinf(min_area):
            return -1.0
            
        return min_area / hull_area

    def objective(params):
        return -compute_score(params)

    # Optimization
    # Bounds for r: (0.1, 0.99)
    # Bounds for theta: (0, 2*pi/7) due to symmetry of heptagon
    bounds = [(0.1, 0.99), (0, 2 * np.pi / 7)]
    
    # Initial guess
    x0 = np.array([0.5, np.pi / 7])
    
    best_result = None
    best_score = -1.0
    
    # Try a few random restarts to avoid local minima, but Nelder-Mead is usually okay for this
    # We can also try L-BFGS-B if we smooth the objective, but Nelder-Mead is safer for 'min' functions.
    
    try:
        # Run Nelder-Mead
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
        
        if res.success and res.fun < best_score:
            best_score = -res.fun
            best_result = res.x
            
        # Check time
        if time.time() > deadline:
            # Return best found so far
            pass
        else:
            # Try a second run with different initial guess if time permits
            # The function is likely unimodal or has few minima.
            x0_2 = np.array([0.7, 0.0])
            res2 = minimize(objective, x0_2, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res2.success and res2.fun < best_score:
                best_score = -res2.fun
                best_result = res2.x
                
    except Exception as e:
        pass

    if best_result is not None:
        r, theta = best_result
        pts = generate_points(r, theta)
        return pts
    else:
        # Fallback: Regular heptagon + regular heptagon at r=0.5, theta=pi/7
        pts = generate_points(0.5, np.pi / 7)
        return pts
