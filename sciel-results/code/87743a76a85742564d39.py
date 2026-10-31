import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 11
H = np.sqrt(3) / 2
SQRT3 = np.sqrt(3)

# Precompute indices for all triangles (triplets)
tri_indices = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            tri_indices.append((i, j, k))
tri_indices = np.array(tri_indices)

def get_points(flat_params):
    return flat_params.reshape(N, 2)

def is_inside(points):
    x = points[:, 0]
    y = points[:, 1]
    
    if np.any(x < -1e-9) or np.any(x > 1 + 1e-9) or np.any(y < -1e-9) or np.any(y > H + 1e-9):
        return False
        
    if np.any(y > SQRT3 * x + 1e-9):
        return False
        
    if np.any(y > SQRT3 * (1 - x) + 1e-9):
        return False
        
    return True

def objective(flat_params):
    points = get_points(flat_params)
    
    if not is_inside(points):
        return 1e6
        
    p1 = points[tri_indices[:, 0]]
    p2 = points[tri_indices[:, 1]]
    p3 = points[tri_indices[:, 2]]
    
    dx1 = p2[:, 0] - p1[:, 0]
    dy1 = p2[:, 1] - p1[:, 1]
    dx2 = p3[:, 0] - p1[:, 0]
    dy2 = p3[:, 1] - p1[:, 1]
    
    cross = dx1 * dy2 - dx2 * dy1
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return -min_area

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    bounds = [(0, 1)] * N + [(0, H)] * N
    
    try:
        de_result = differential_evolution(
            objective,
            bounds,
            maxiter=500,
            popsize=20,
            tol=1e-6,
            seed=42,
            polish=False,
            init='sobol'
        )
    except Exception:
        # Fallback
        rng = np.random.RandomState(42)
        while True:
            pts = np.random.rand(N, 2)
            pts[:, 0] *= 1
            pts[:, 1] *= H
            if is_inside(pts):
                break
        return pts

    best_params = de_result.x
    
    elapsed = time.time() - start_time
    if elapsed < time_limit:
        try:
            res = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if res.fun < de_result.fun:
                best_params = res.x
        except Exception:
            pass
            
    points = get_points(best_params)
    
    if not is_inside(points):
        # Fallback
        rng = np.random.RandomState(42)
        while True:
            pts = np.random.rand(N, 2)
            pts[:, 0] *= 1
            pts[:, 1] *= H
            if is_inside(pts):
                break
        return pts
        
    return points
