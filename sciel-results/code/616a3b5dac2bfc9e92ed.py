import numpy as np
import time
from scipy.optimize import minimize

N = 11
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3) / 2.0])

def bary_to_cart(u, v):
    x = u + 0.5 * v
    y = (np.sqrt(3) / 2.0) * v
    return np.column_stack((x, y))

def min_triangle_area(points):
    n = len(points)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                v1 = p2 - p1
                v2 = p3 - p1
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def objective(x):
    u = x[0::2]
    v = x[1::2]
    
    # Penalty for invalid points
    pen = 0.0
    pen += np.sum(np.maximum(0, -u) ** 2)
    pen += np.sum(np.maximum(0, -v) ** 2)
    pen += np.sum(np.maximum(0, u + v - 1) ** 2)
    
    if pen > 1e-8:
        return 1e10 + pen

    u_v = np.column_stack((u, v))
    points = bary_to_cart(u_v)
    
    area = min_triangle_area(points)
    return -area

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_score = -np.inf
    best_x = None
    
    bounds = [(0, 1), (0, 1)] * N
    
    np.random.seed(42)
    
    # Multi-start
    for i in range(50):
        if time.time() > deadline:
            break
        
        u = np.random.rand(N)
        v = np.random.rand(N)
        for j in range(N):
            if u[j] + v[j] > 1:
                u[j], v[j] = 1 - u[j], 1 - v[j]
        
        x0 = np.empty(2 * N)
        x0[0::2] = u
        x0[1::2] = v
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500})
            
            if res.fun < best_score:
                u_opt = res.x[0::2]
                v_opt = res.x[1::2]
                
                if np.all(u_opt >= -1e-6) and np.all(v_opt >= -1e-6) and np.all(u_opt + v_opt <= 1 + 1e-6):
                    u_v = np.column_stack((u_opt, v_opt))
                    points = bary_to_cart(u_v)
                    score = min_triangle_area(points)
                    
                    if score > best_score:
                        best_score = score
                        best_x = res.x
        except Exception:
            continue
            
    if best_x is None:
        u = np.random.rand(N)
        v = np.random.rand(N)
        for j in range(N):
            if u[j] + v[j] > 1:
                u[j], v[j] = 1 - u[j], 1 - v[j]
        best_x = np.empty(2 * N)
        best_x[0::2] = u
        best_x[1::2] = v
        
    u_opt = best_x[0::2]
    v_opt = best_x[1::2]
    u_v = np.column_stack((u_opt, v_opt))
    points = bary_to_cart(u_v)
    
    return points
