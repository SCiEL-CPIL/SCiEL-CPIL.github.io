import numpy as np
import itertools
import time
from scipy.optimize import minimize

N = 11
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3) / 2])
tri_indices = list(itertools.combinations(range(N), 3))

def get_areas(points):
    areas = np.zeros(len(tri_indices))
    for idx, (i, j, k) in enumerate(tri_indices):
        p1 = points[i]
        p2 = points[j]
        p3 = points[k]
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        areas[idx] = 0.5 * abs(cross)
    return areas

def objective_smoothed(x, beta=100.0):
    points = x.reshape((N, 2))
    areas = get_areas(points)
    exponents = -beta * areas
    max_exp = np.max(exponents)
    shifted = exponents - max_exp
    log_sum = max_exp + np.log(np.sum(np.exp(shifted)))
    return log_sum / beta

def constraints_func(x):
    points = x.reshape((N, 2))
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    
    c1 = y_coords
    c2 = np.sqrt(3) * x_coords - y_coords
    c3 = -np.sqrt(3) * (x_coords - 1) - y_coords
    
    return np.concatenate([c1, c2, c3])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_points = None
    best_score = -np.inf
    
    # Use more restarts
    n_restarts = 2000
    
    for i in range(n_restarts):
        if time.time() > deadline:
            break
            
        # Random initialization
        pts = []
        for _ in range(N):
            while True:
                u = np.random.rand()
                v = np.random.rand()
                if u + v < 1:
                    break
            p = u * B + v * C
            pts.append(p)
        x0 = np.array(pts).flatten()
        
        try:
            res = minimize(
                objective_smoothed, 
                x0, 
                method='SLSQP', 
                constraints=[{'type': 'ineq', 'fun': constraints_func}],
                options={'maxiter': 100, 'ftol': 1e-6}
            )
            
            if res.success:
                x_opt = res.x
                points_opt = x_opt.reshape((N, 2))
                
                # Verify constraints are satisfied
                cons = constraints_func(x_opt)
                if np.min(cons) >= -1e-5:
                    areas = get_areas(points_opt)
                    min_area = np.min(areas)
                    
                    if min_area > best_score:
                        best_score = min_area
                        best_points = points_opt.copy()
                        
        except Exception:
            continue
            
    if best_points is None:
        pts = []
        for _ in range(N):
            while True:
                u = np.random.rand()
                v = np.random.rand()
                if u + v < 1:
                    break
            p = u * B + v * C
            pts.append(p)
        best_points = np.array(pts)
        
    return best_points
