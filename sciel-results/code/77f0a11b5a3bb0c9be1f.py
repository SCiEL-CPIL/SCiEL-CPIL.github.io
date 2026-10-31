import numpy as np
from scipy.optimize import minimize
import time

N = 11
SQRT3 = np.sqrt(3)

# Precompute indices for triangle combinations
_idx_arr = np.array(list(zip(*np.ndindex(N, N, N))))
_mask = (_idx_arr[:, 0] < _idx_arr[:, 1]) & (_idx_arr[:, 1] < _idx_arr[:, 2])
_IDX = _idx_arr[_mask]

def vectorized_triangle_areas(points):
    p1 = points[_IDX[:, 0]]
    p2 = points[_IDX[:, 1]]
    p3 = points[_IDX[:, 2]]
    
    areas = 0.5 * np.abs(
        p1[:, 0] * (p2[:, 1] - p3[:, 1]) + 
        p2[:, 0] * (p3[:, 1] - p1[:, 1]) + 
        p3[:, 0] * (p1[:, 1] - p2[:, 1])
    )
    return areas

def penalty(points):
    x = points[:, 0]
    y = points[:, 1]
    d1 = y
    d2 = SQRT3 * x - y
    d3 = SQRT3 - SQRT3 * x - y
    pen = np.sum(np.maximum(0, -d1)**2) + np.sum(np.maximum(0, -d2)**2) + np.sum(np.maximum(0, -d3)**2)
    return pen

def smooth_min_area(points, temp):
    areas = vectorized_triangle_areas(points)
    min_val = np.min(areas)
    if min_val < 1e-12:
        return 0.0
    log_sum = np.log(np.sum(np.exp(-(areas - min_val) / temp)))
    return min_val - temp * log_sum

def objective_smooth(flat_params, temp):
    points = flat_params.reshape(N, 2)
    if np.any(points[:, 0] < -0.5) or np.any(points[:, 0] > 1.5) or np.any(points[:, 1] < -0.5) or np.any(points[:, 1] > 1.5):
        return 1e6
    min_area_approx = smooth_min_area(points, temp)
    pen = penalty(points)
    return -min_area_approx + 50.0 * pen

def objective_exact(flat_params):
    points = flat_params.reshape(N, 2)
    if np.any(points[:, 0] < -0.5) or np.any(points[:, 0] > 1.5) or np.any(points[:, 1] < -0.5) or np.any(points[:, 1] > 1.5):
        return 1e6
    areas = vectorized_triangle_areas(points)
    min_area = np.min(areas)
    pen = penalty(points)
    return -min_area + 1000.0 * pen

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    best_points = None
    best_score = -np.inf
    
    inits = []
    for seed in range(20):
        rng = np.random.RandomState(seed)
        init_points = np.zeros((N, 2))
        for i in range(N):
            while True:
                u = rng.rand()
                v = rng.rand()
                if u + v <= 1:
                    break
            w = 1 - u - v
            init_points[i, 0] = v * 1.0 + w * 0.5
            init_points[i, 1] = w * SQRT3 / 2.0
        inits.append(init_points)
        
    for init in inits:
        if time.time() - start_time > time_limit:
            break
            
        flat_init = init.flatten()
        
        try:
            res1 = minimize(
                objective_smooth,
                flat_init,
                args=(0.02,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            points1 = res1.x.reshape(N, 2)
            
            res2 = minimize(
                objective_exact,
                points1.flatten(),
                method='Powell',
                options={'maxiter': 100, 'maxfev': 5000}
            )
            points2 = res2.x.reshape(N, 2)
            
            min_area = np.min(vectorized_triangle_areas(points2))
            pen = penalty(points2)
            score = min_area - 1000 * pen
            
            if pen < 1e-4 and score > best_score:
                best_score = score
                best_points = points2.copy()
                
        except Exception:
            pass

    if best_points is None:
        return np.array([[0.5, SQRT3/6]] * N)
        
    return best_points
