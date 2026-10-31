import numpy as np
from scipy.optimize import minimize

N = 11
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3)/2])

def point_in_triangle(p, a, b, c):
    v0 = c - a
    v1 = b - a
    v2 = p - a
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    if denom == 0:
        return False
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    return (v >= 0) and (w >= 0) and (v + w <= 1)

def min_triangle_area_fast(points):
    min_area = np.inf
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i+1, n):
            x2, y2 = points[j]
            for k in range(j+1, n):
                x3, y3 = points[k]
                val = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                if val < min_area:
                    min_area = val
    return min_area * 0.5

def objective(params):
    points = params.reshape(N, 2)
    for p in points:
        if not point_in_triangle(p, A, B, C):
            return 1000.0
    return -min_triangle_area_fast(points)

def construct():
    best_params = None
    best_val = -np.inf
    rng = np.random.default_rng(42)
    
    for _ in range(30):
        u = np.sqrt(rng.random(N))
        v = rng.random(N) * (1 - u)
        
        # Add small margin to avoid boundary issues during optimization
        # Scale u, v to [0.1, 0.9] range roughly
        # Actually, just use the mapped points
        x = u * 1.0 + v * 0.5
        y = v * (np.sqrt(3)/2)
        points = np.column_stack((x, y))
        
        res = minimize(objective, points.flatten(), method='Nelder-Mead', 
                       options={'maxiter': 1500, 'xatol': 1e-7, 'fatol': 1e-7})
        
        if res.fun < best_val:
            final_points = res.x.reshape(N, 2)
            valid = True
            for p in final_points:
                if not point_in_triangle(p, A, B, C):
                    valid = False
                    break
            if valid:
                best_val = res.fun
                best_params = res.x

    if best_params is None:
        return np.array([[rng.random(), rng.random()*np.sqrt(3)/2] for _ in range(N)])
        
    return best_params.reshape(N, 2)
