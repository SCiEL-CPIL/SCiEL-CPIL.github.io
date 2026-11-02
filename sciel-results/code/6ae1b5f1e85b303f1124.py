import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import itertools

N = 11
H = np.sqrt(3) / 2
SQRT3 = np.sqrt(3)

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin for 150s limit
    
    # Precompute indices for combinations
    indices = list(itertools.combinations(range(N), 3))
    
    def check_validity(points):
        x = points[:, 0]
        y = points[:, 1]
        if np.any(y < -1e-9): return False
        if np.any(y > SQRT3 * x + 1e-9): return False
        if np.any(y > -SQRT3 * x + SQRT3 + 1e-9): return False
        return True

    def min_area(points):
        x = points[:, 0]
        y = points[:, 1]
        min_area = float('inf')
        for i, j, k in indices:
            area = 0.5 * abs(x[i]*(y[j]-y[k]) + x[j]*(y[k]-y[i]) + x[k]*(y[i]-y[j]))
            if area < min_area:
                min_area = area
        return min_area

    def objective_de(params):
        points = params.reshape(N, 2)
        if not check_validity(points):
            return 1e10
        ma = min_area(points)
        if ma < 1e-12:
            return 1e10
        return -ma

    # 1. Global Search with Differential Evolution
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, H))   # y
    
    result_de = differential_evolution(
        objective_de, 
        bounds, 
        maxiter=200, 
        popsize=20, 
        tol=1e-6, 
        seed=42, 
        mutation=(0.5, 1.5), 
        recombination=0.9,
        polish=False
    )
    
    best_points = result_de.x.reshape(N, 2)
    best_score = -result_de.fun
    
    elapsed = time.time() - start_time
    if elapsed > timeout - 30:
        if not check_validity(best_points):
             return _random_valid_points(N)
        return best_points

    # 2. Local Refinement with Soft-Min Objective
    # Use log-sum-exp approximation for the minimum to make it differentiable
    # min(a_i) ≈ -T * log(sum(exp(-a_i/T)))
    # We want to maximize min(a_i), so we minimize -min(a_i)
    # Objective: T * log(sum(exp(-a_i/T))) + penalty
    
    def objective_softmin(params, T=0.01):
        points = params.reshape(N, 2)
        x = points[:, 0]
        y = points[:, 1]
        
        # Penalty for invalidity
        pen = 0
        if np.any(y < 0):
            pen += np.sum(np.maximum(0, -y)**2)
        if np.any(y > SQRT3 * x):
            pen += np.sum(np.maximum(0, y - SQRT3*x)**2)
        if np.any(y > -SQRT3 * x + SQRT3):
            pen += np.sum(np.maximum(0, y - (-SQRT3*x + SQRT3))**2)
        
        if pen > 1e-6:
            return 1e10 + pen

        # Calculate all areas
        areas = np.zeros(len(indices))
        for idx, (i, j, k) in enumerate(indices):
            areas[idx] = 0.5 * abs(x[i]*(y[j]-y[k]) + x[j]*(y[k]-y[i]) + x[k]*(y[i]-y[j]))
        
        # Soft min
        # To avoid overflow in exp, we can shift, but areas are small so it's usually fine.
        # Let's clamp areas to a small positive number to avoid log(0) or exp(-inf)
        areas_clipped = np.maximum(areas, 1e-12)
        log_sum_exp = np.log(np.sum(np.exp(-areas_clipped / T)))
        soft_min = -T * log_sum_exp
        
        return -soft_min

    # Polish with L-BFGS-B using the soft-min objective
    try:
        res_lbfgs = minimize(
            objective_softmin, 
            best_points.flatten(), 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if res_lbfgs.fun < 1e9: # Check if it didn't blow up
            new_points = res_lbfgs.x.reshape(N, 2)
            if check_validity(new_points):
                new_score = min_area(new_points)
                if new_score > best_score:
                    best_points = new_points
                    best_score = new_score
    except Exception:
        pass

    elapsed = time.time() - start_time
    if elapsed > timeout - 15:
        if not check_validity(best_points):
             return _random_valid_points(N)
        return best_points

    # 3. Multi-start with Soft-Min Polish
    # Try a few random restarts, polish with soft-min, keep best
    num_restarts = 3
    for i in range(num_restarts):
        if time.time() - start_time > timeout - 5:
            break
            
        pts = _random_valid_points(N)
        
        try:
            res_r = minimize(
                objective_softmin, 
                pts.flatten(), 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res_r.fun < 1e9:
                 if check_validity(res_r.x.reshape(N, 2)):
                     score_r = min_area(res_r.x.reshape(N, 2))
                     if score_r > best_score:
                         best_points = res_r.x.reshape(N, 2)
                         best_score = score_r
        except Exception:
            pass

    if not check_validity(best_points):
        return _random_valid_points(N)
        
    return best_points

def _random_valid_points(n):
    points = []
    for _ in range(n):
        while True:
            u, v = np.random.rand(2)
            if u + v < 1:
                # Barycentric coordinates
                # V0=(0,0), V1=(1,0), V2=(0.5, H)
                # P = u*V0 + v*V1 + (1-u-v)*V2
                x = v * 1.0 + (1-u-v) * 0.5
                y = (1-u-v) * H
                points.append([x, y])
                break
    return np.array(points)
