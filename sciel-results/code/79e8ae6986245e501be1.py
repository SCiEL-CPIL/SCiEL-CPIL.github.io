import numpy as np
import time
import itertools
from scipy.optimize import differential_evolution, minimize

N = 11

# Precompute indices for all triangles
_indices = np.array(list(itertools.combinations(range(N), 3)))

def decode_points(uv):
    u = uv[:, 0]
    v = uv[:, 1]
    s = u + v
    mask = s > 1.0
    u_new = u.copy()
    v_new = v.copy()
    # Handle s==0 case (origin)
    # If s==0, u=0, v=0, point is (0,0).
    # Division by zero is avoided by mask logic if we are careful, 
    # but s>1 is false for s=0, so u_new=u, v_new=v.
    if np.any(mask):
        u_new[mask] = u[mask] / s[mask]
        v_new[mask] = v[mask] / s[mask]
    
    x = u_new + 0.5 * v_new
    y = (np.sqrt(3) / 2) * v_new
    return np.column_stack((x, y))

def min_triangle_area(points):
    p1 = points[_indices[:, 0]]
    p2 = points[_indices[:, 1]]
    p3 = points[_indices[:, 2]]
    areas = 0.5 * np.abs(p1[:, 0]*(p2[:, 1]-p3[:, 1]) + p2[:, 0]*(p3[:, 1]-p1[:, 1]) + p3[:, 0]*(p1[:, 1]-p2[:, 1]))
    return np.min(areas)

def objective(x_flat):
    uv = x_flat.reshape(N, 2)
    uv = np.clip(uv, 0, 1)
    points = decode_points(uv)
    area = min_triangle_area(points)
    return -area

def construct():
    start_time = time.time()
    time_limit = 140.0
    
    bounds = [(0, 1)] * (2 * N)
    
    best_result = None
    best_val = -np.inf
    
    # Global Search
    try:
        res_de = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=10, 
            tol=1e-10,
            polish=False
        )
        best_result = res_de.x
        best_val = res_de.fun
    except Exception:
        best_result = np.random.rand(2 * N)
        best_val = objective(best_result)

    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    
    # Local Polish
    if remaining > 5:
        try:
            res_local = minimize(
                objective, 
                best_result, 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'maxfev': 20000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if res_local.fun < best_val:
                best_val = res_local.fun
                best_result = res_local.x
        except Exception:
            pass

    uv = np.clip(best_result.reshape(N, 2), 0, 1)
    points = decode_points(uv)
    return points
