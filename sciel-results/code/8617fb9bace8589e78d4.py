import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 145

def _min_area(points):
    """Vectorized min area calculation."""
    n = len(points)
    # Generate indices for all unique triplets (i, j, k) with i < j < k
    i, j, k = np.indices((n, n, n))
    mask = (i < j) & (j < k)
    
    pi = points[i[mask]]
    pj = points[j[mask]]
    pk = points[k[mask]]
    
    # Cross product magnitude
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _penalized_min_area(flat_params):
    points = flat_params.reshape(-1, 2)
    x = points[:, 0]
    y = points[:, 1]
    
    # Vectorized penalty for triangle boundaries
    # y >= 0, y <= S3H
    # x >= 0, x <= 1
    # y <= sqrt(3) * x
    # y <= sqrt(3) * (1-x)
    
    pen = 0.0
    pen += np.sum(np.maximum(0, -y)**2)
    pen += np.sum(np.maximum(0, y - S3H)**2)
    pen += np.sum(np.maximum(0, -x)**2)
    pen += np.sum(np.maximum(0, x - 1)**2)
    
    left_viol = np.maximum(0, y - np.sqrt(3) * x)
    right_viol = np.maximum(0, y - np.sqrt(3) * (1 - x))
    pen += np.sum(left_viol**2) + np.sum(right_viol**2)
    
    ma = _min_area(points)
    return -ma + pen

def construct():
    start_time = time.time()
    
    bounds = [(0, 1), (0, S3H)] * N
    
    best_flat = None
    best_val = float('inf')
    
    # Run DE with more iterations because _min_area is now fast
    seeds = [42, 123, 456, 789, 321, 111, 222, 333]
    
    for seed in seeds:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 20: # Leave 20s for polishing
            break
            
        try:
            result = differential_evolution(
                _penalized_min_area,
                bounds,
                maxiter=100,
                popsize=30,
                seed=seed,
                tol=1e-6,
                polish=False,
                init='latinhypercube'
            )
            
            # Check validity roughly
            pts = result.x.reshape(-1, 2)
            if np.all(pts[:, 1] >= 0) and np.all(pts[:, 1] <= S3H) and np.all(pts[:, 0] >= 0) and np.all(pts[:, 0] <= 1):
                if np.all(pts[:, 1] <= np.sqrt(3) * pts[:, 0]) and np.all(pts[:, 1] <= np.sqrt(3) * (1 - pts[:, 0])):
                    if result.fun < best_val:
                        best_val = result.fun
                        best_flat = result.x.copy()
        except Exception:
            pass
            
    if best_flat is None:
        # Fallback: return a valid random config
        # Barycentric random
        u = np.random.rand(N)
        v = np.random.rand(N)
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        v1 = np.array([0.0, 0.0])
        v2 = np.array([1.0, 0.0])
        v3 = np.array([0.5, S3H])
        pts = (1 - u - v)[:, np.newaxis] * v1 + u[:, np.newaxis] * v2 + v[:, np.newaxis] * v3
        return pts

    # Local Polish
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5:
        try:
            # Nelder-Mead
            result = minimize(
                _penalized_min_area,
                best_flat,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_flat = result.x.copy()
        except Exception:
            pass
            
    # Second Polish with L-BFGS-B
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 2:
        try:
            result = minimize(
                _penalized_min_area,
                best_flat,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_flat = result.x.copy()
        except Exception:
            pass

    points = best_flat.reshape(-1, 2)
    
    # Final Projection to ensure validity
    for i in range(len(points)):
        x, y = points[i]
        y = max(0, min(S3H, y))
        x_min = y / np.sqrt(3)
        x_max = 1 - y / np.sqrt(3)
        x = max(x_min, min(x_max, x))
        points[i] = [x, y]
        
    return points
