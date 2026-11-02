import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 11
H = np.sqrt(3) / 2

def is_inside_triangle(points):
    # points: (N, 2)
    x = points[:, 0]
    y = points[:, 1]
    # y >= 0
    if np.any(y < -1e-9): return False
    # y <= sqrt(3) * x
    if np.any(y > np.sqrt(3) * x + 1e-9): return False
    # y <= sqrt(3) * (1 - x)
    if np.any(y > np.sqrt(3) * (1 - x) + 1e-9): return False
    return True

def min_triangle_area(points):
    # points: (N, 2)
    n = points.shape[0]
    min_area = float('inf')
    # Iterate over all triplets
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def objective(x):
    points = x.reshape(N, 2)
    if not is_inside_triangle(points):
        return 1e6 # Penalty
    return -min_triangle_area(points)

def construct():
    start_time = time.time()
    deadline = start_time + 110 # 10s safety margin
    
    # Bounds for DE
    # x in [0, 1], y in [0, H]
    # We restrict slightly to avoid boundary issues in DE
    bounds = []
    for _ in range(N):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, H - 0.01))
        
    best_x = None
    best_val = -1e6
    
    # Run DE
    # maxiter and popsize control the budget
    # 22 variables. popsize=10, maxiter=50 -> ~500 evals.
    # We can run multiple DE instances with different seeds if time permits.
    
    def run_de(seed):
        nonlocal best_x, best_val
        if time.time() > deadline:
            return
        try:
            res = differential_evolution(
                objective, 
                bounds, 
                seed=seed, 
                maxiter=30, 
                popsize=10, 
                tol=1e-4,
                polish=False # We will polish manually
            )
            if time.time() > deadline:
                return
            if not res.success:
                # Still check the result
                pass
            val = -res.fun
            if val > best_val:
                best_val = val
                best_x = res.x
        except Exception:
            pass

    # Run a few DE instances
    for seed in [1, 2, 3]:
        if time.time() > deadline - 10: # Leave 10s for polishing
            break
        run_de(seed)
        
    if best_x is None:
        # Fallback to random
        rng = np.random.RandomState(0)
        points = []
        for _ in range(N):
            u, v = rng.random(), rng.random()
            while u + v > 1:
                u, v = rng.random(), rng.random()
            x = 0.5 * u + 0.5 * (1-u) * 0.5 + 0.5 * (1-u) * 0.5 # This mapping is wrong
            # Simple random in triangle:
            u, v = rng.random(), rng.random()
            if u + v > 1:
                u, v = 1-u, 1-v
            x = u * 0.5 + v * 0.5
            y = v * H
            points.append([x, y])
        return np.array(points)

    # Polish with Nelder-Mead
    if time.time() < deadline - 5:
        try:
            res_nm = minimize(
                objective, 
                best_x, 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if time.time() < deadline:
                val_nm = -res_nm.fun
                if val_nm > best_val:
                    # Check validity
                    pts = res_nm.x.reshape(N, 2)
                    if is_inside_triangle(pts):
                        best_x = res_nm.x
                        best_val = val_nm
        except Exception:
            pass

    return best_x.reshape(N, 2)
