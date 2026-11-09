import numpy as np
import time
from scipy.optimize import minimize

def _is_feasible(centers, sides, tol=1e-7):
    n = len(sides)
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if s <= 0 or not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(s)):
            return False
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j])/2 - tol
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            if dx < t and dy < t:
                return False
    return True

def _penalty(centers, sides, tol=1e-7, weight=1e4):
    n = len(sides)
    pen = 0.0
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if s <= 0:
            pen += weight * (1.0 - s)
        if cx - s/2 < -tol:
            pen += weight * (-tol - (cx - s/2))
        if cx + s/2 > 1 + tol:
            pen += weight * ((cx + s/2) - (1 + tol))
        if cy - s/2 < -tol:
            pen += weight * (-tol - (cy - s/2))
        if cy + s/2 > 1 + tol:
            pen += weight * ((cy + s/2) - (1 + tol))
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j])/2 - tol
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            if dx < t and dy < t:
                pen += weight * ((t - dx) * (t - dy))
    return pen

def _objective(x, tol=1e-7, weight=1e4):
    n = 10
    centers = x[:2*n].reshape(n, 2)
    sides = x[2*n:]
    return -np.sum(sides) + _penalty(centers, sides, tol, weight)

def _local_optimize(centers, sides, max_iter=1000, tol=1e-7):
    n = len(sides)
    x0 = np.concatenate([centers.flatten(), sides])
    res = minimize(_objective, x0, args=(tol, 1e4), method='Nelder-Mead', 
                   options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8})
    x_opt = res.x
    centers_opt = x_opt[:2*n].reshape(n, 2)
    sides_opt = x_opt[2*n:]
    return centers_opt, sides_opt

def squares_in_square10() -> tuple:
    start_time = time.time()
    deadline = start_time + 580
    
    best_centers = None
    best_sides = None
    best_score = -np.inf
    
    # Initial config 1: 2 big, 8 small
    c1 = np.array([[0.25, 0.75], [0.75, 0.75]] + [[(i+0.5)*0.25, 0.375] for i in range(4)] + [[(i+0.5)*0.25, 0.125] for i in range(4)])
    s1 = np.array([0.5, 0.5] + [0.25]*8)
    
    # Initial config 2: 4 medium, 6 small
    c2 = np.array([[0.25, 0.75], [0.75, 0.75], [0.25, 0.25], [0.75, 0.25]] + [[0.125, 0.5], [0.375, 0.5], [0.625, 0.5], [0.875, 0.5], [0.125, 0.5], [0.375, 0.5]])
    s2 = np.array([0.4]*4 + [0.2]*6)
    
    configs = [(c1, s1), (c2, s2)]
    
    # Add random configs
    np.random.seed(42)
    for _ in range(5):
        c_rand = np.random.rand(10, 2)
        s_rand = np.random.rand(10) * 0.2 + 0.1
        configs.append((c_rand, s_rand))
        
    for c, s in configs:
        if time.time() > deadline:
            break
        c_opt, s_opt = _local_optimize(c, s, max_iter=500)
        if _is_feasible(c_opt, s_opt):
            score = np.sum(s_opt)
            if score > best_score:
                best_score = score
                best_centers = c_opt
                best_sides = s_opt
                
    if best_centers is None:
        # Fallback
        c_fb = np.array([[0.25, 0.75], [0.75, 0.75]] + [[(i+0.5)*0.25, 0.375] for i in range(4)] + [[(i+0.5)*0.25, 0.125] for i in range(4)])
        s_fb = np.array([0.5, 0.5] + [0.25]*8)
        return c_fb, s_fb
        
    return best_centers, best_sides
