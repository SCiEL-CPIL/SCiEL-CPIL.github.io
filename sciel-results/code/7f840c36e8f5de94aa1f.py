import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

N = 11
_H = np.sqrt(3) / 2.0

# Precompute indices
combos = list(combinations(range(N), 3))
idx_i = np.array([c[0] for c in combos])
idx_j = np.array([c[1] for c in combos])
idx_k = np.array([c[2] for c in combos])

def _cartesian_from_bary(z):
    u = z[:, 0]
    v = z[:, 1]
    x = u + 0.5 * v
    y = _H * v
    return np.column_stack((x, y))

def _compute_areas_from_bary(z):
    pts = _cartesian_from_bary(z)
    pi = pts[idx_i]
    pj = pts[idx_j]
    pk = pts[idx_k]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    return 0.5 * np.abs(cross)

def _objective_true_bary(z):
    areas = _compute_areas_from_bary(z)
    return -np.min(areas)

def _objective_soft_min_bary(z, beta):
    areas = _compute_areas_from_bary(z)
    m = np.min(areas)
    exp_terms = np.exp(-beta * (areas - m))
    log_sum = np.log(np.sum(exp_terms))
    soft_min_val = m - log_sum / beta
    return -soft_min_val

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_z = None
    best_score = -np.inf
    
    rng = np.random.RandomState(42)
    
    cons = []
    for i in range(N):
        def constraint_func(z, i=i):
            return 1.0 - z[i, 0] - z[i, 1]
        cons.append({'type': 'ineq', 'fun': constraint_func})
        
    bounds = [(0.0, 1.0) for _ in range(2*N)]
    
    # Try multiple starts
    for start_idx in range(15):
        if time.time() > deadline:
            break
            
        u = rng.random(N)
        v = rng.random(N)
        while np.any(u + v > 1.0):
            mask = u + v > 1.0
            u[mask] = rng.random(np.sum(mask))
            v[mask] = rng.random(np.sum(mask))
            
        z0 = np.column_stack((u, v))
        
        try:
            res = minimize(
                _objective_soft_min_bary,
                z0,
                args=(50.0,),
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 200, 'ftol': 1e-10}
            )
        except Exception:
            continue
            
        if time.time() > deadline:
            if best_z is None:
                best_z = res.x
                best_score = _objective_true_bary(best_z)
            break
            
        z_opt = res.x
        
        try:
            res2 = minimize(
                _objective_soft_min_bary,
                z_opt,
                args=(200.0,),
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            z_opt = res2.x
        except Exception:
            pass
            
        score = _objective_true_bary(z_opt)
        
        if score > best_score:
            best_score = score
            best_z = z_opt.copy()
            
    if best_z is None:
        u = np.ones(N) / (N+1)
        v = np.ones(N) / (N+1)
        best_z = np.column_stack((u, v))
        
    best_points = _cartesian_from_bary(best_z)
    
    return best_points
