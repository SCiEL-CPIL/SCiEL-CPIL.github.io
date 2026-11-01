import numpy as np
from scipy.optimize import minimize
import time

N = 11

# Vertices of a unit regular hexagon with angle 0
HEX_VERTS = np.array([
    [1.0, 0.0],
    [0.5, np.sqrt(3)/2],
    [-0.5, np.sqrt(3)/2],
    [-1.0, 0.0],
    [-0.5, -np.sqrt(3)/2],
    [0.5, -np.sqrt(3)/2]
])

def s_req(x, y):
    """
    Calculate the minimum side length S of a regular hexagon (angle 0, center 0)
    that contains the point (x, y).
    """
    term1 = np.abs(y) * 2.0 / np.sqrt(3.0)
    term2 = np.abs(x + y / np.sqrt(3.0))
    term3 = np.abs(x - y / np.sqrt(3.0))
    return np.maximum(term1, np.maximum(term2, term3))

def construct():
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    sqrt3 = np.sqrt(3)
    
    # Initial guess: 3-4-3-1 cluster
    init_points = []
    init_points.extend([[-sqrt3, 0.0], [0.0, 0.0], [sqrt3, 0.0]])
    init_points.extend([[-1.5*sqrt3, 1.5], [-0.5*sqrt3, 1.5], [0.5*sqrt3, 1.5], [1.5*sqrt3, 1.5]])
    init_points.extend([[-sqrt3, 3.0], [0.0, 3.0], [sqrt3, 3.0]])
    init_points.extend([[0.0, 4.5]])
    
    # Center the cluster
    cx = np.mean([p[0] for p in init_points])
    cy = np.mean([p[1] for p in init_points])
    init_points = [[p[0]-cx, p[1]-cy] for p in init_points]
    
    x0 = np.array([coord for p in init_points for coord in p])
    
    def obj_func(params):
        s_max = 0.0
        for i in range(N):
            xi = params[2*i]
            yi = params[2*i+1]
            for v in HEX_VERTS:
                vx = xi + v[0]
                vy = yi + v[1]
                s_val = s_req(vx, vy)
                if s_val > s_max:
                    s_max = s_val
        return s_max

    def obj_with_penalty(params):
        s_max = obj_func(params)
        penalty = 0.0
        for i in range(N):
            xi = params[2*i]
            yi = params[2*i+1]
            for j in range(i+1, N):
                xj = params[2*j]
                yj = params[2*j+1]
                d2 = (xi-xj)**2 + (yi-yj)**2
                if d2 < 3.0:
                    penalty += (3.0 - d2)**2
        return s_max + 1000.0 * penalty

    # Use Nelder-Mead
    res = minimize(obj_with_penalty, x0, method='Nelder-Mead', options={'maxiter': 20000, 'xatol': 1e-8, 'fatol': 1e-8})
    
    if time.time() > deadline:
        # If it times out, we can't easily stop minimize.
        # But Nelder-Mead is fast. 20000 iterations on 22 vars is likely < 10s.
        pass

    best_params = res.x
    best_score = obj_func(best_params)
    
    # Verify disjointness
    valid = True
    for i in range(N):
        xi = best_params[2*i]
        yi = best_params[2*i+1]
        for j in range(i+1, N):
            xj = best_params[2*j]
            yj = best_params[2*j+1]
            d2 = (xi-xj)**2 + (yi-yj)**2
            if d2 < 3.0 - 1e-6:
                valid = False
                break
        if not valid: break
        
    if not valid:
        # Fallback
        best_params = x0
        best_score = obj_func(x0)
        # The initial lattice configuration is valid.
        
    inner = [[best_params[2*i], best_params[2*i+1], 0.0] for i in range(N)]
    return inner, (0.0, 0.0), best_score, 0.0
