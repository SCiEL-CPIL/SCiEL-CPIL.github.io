import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 11
H = np.sqrt(3) / 2.0

def get_area(p1, p2, p3):
    # p1, p2, p3 are (x, y)
    return 0.5 * abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))

def min_triangle_area(points):
    # points is (N, 2)
    min_area = float('inf')
    # Optimization: Pre-calculate indices
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                a = get_area(points[i], points[j], points[k])
                if a < min_area:
                    min_area = a
    return min_area

def penalty(points):
    # points is (N, 2)
    x = points[:, 0]
    y = points[:, 1]
    # Inside triangle: x >= 0, y >= 0, y <= sqrt(3)*(1-x)
    viol = 0.0
    viol += np.sum(np.maximum(0.0, -x))
    viol += np.sum(np.maximum(0.0, -y))
    viol += np.sum(np.maximum(0.0, y - np.sqrt(3)*(1.0 - x)))
    return viol

def objective(x_flat):
    points = x_flat.reshape(N, 2)
    p = penalty(points)
    if p > 1e-9:
        return 1e6 + p
    area = min_triangle_area(points)
    return -area

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    bounds = []
    for _ in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, H))
        
    best_result = None
    best_val = -1.0
    
    # Global Search
    try:
        def callback(xk, convergence):
            nonlocal best_result, best_val
            val = objective(xk)
            if val < best_val:
                best_val = val
                best_result = xk.copy()
            return time.time() > deadline

        # DE parameters:
        # popsize=15 is default. 
        # maxiter is high, but callback stops it.
        # tol=1e-10
        res = differential_evolution(objective, bounds, maxiter=10000, tol=1e-12, 
                                      polish=False, callback=callback, 
                                      seed=42, maxfev=50000)
        
        if res.x is not None:
            val = objective(res.x)
            if val < best_val:
                best_val = val
                best_result = res.x.copy()
                
    except Exception:
        pass

    # Local Refinement
    if best_result is not None and time.time() < deadline:
        try:
            # Nelder-Mead is robust for non-smooth functions
            res_local = minimize(objective, best_result, method='Nelder-Mead', 
                                 options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-14})
            if res_local.fun < best_val:
                best_val = res_local.fun
                best_result = res_local.x.copy()
        except Exception:
            pass

    if best_result is None:
        # Fallback
        valid_pts = []
        while len(valid_pts) < N:
            u = np.random.rand()
            v = np.random.rand()
            if u + v < 1:
                x = u
                y = v * H
                valid_pts.append([x, y])
        best_result = np.array(valid_pts).flatten()

    points = best_result.reshape(N, 2)
    
    # Final projection to ensure validity
    x = points[:, 0]
    y = points[:, 1]
    
    for i in range(N):
        if x[i] < 0: x[i] = 0
        if y[i] < 0: y[i] = 0
        if x[i] > 1: x[i] = 1
        if y[i] > H: y[i] = H
        if y[i] > np.sqrt(3) * (1 - x[i]):
            # Move to centroid if outside
            x[i] = 0.5
            y[i] = H/3
            
    points[:, 0] = x
    points[:, 1] = y
    
    return points
