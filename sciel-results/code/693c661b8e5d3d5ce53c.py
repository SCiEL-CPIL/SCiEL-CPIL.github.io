"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 11
H = np.sqrt(3) / 2
SQRT3 = np.sqrt(3)

# Precompute indices for all C(11,3) = 165 triples
_idx_i, _idx_j, _idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _idx_i.append(i)
            _idx_j.append(j)
            _idx_k.append(k)
_idx_i = np.array(_idx_i)
_idx_j = np.array(_idx_j)
_idx_k = np.array(_idx_k)

def _min_area(points):
    """Compute minimum triangle area among all C(N,3) triples. points: (N,2)."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    
    v1 = p2 - p1
    v2 = p3 - p1
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _objective(x):
    """Objective function for minimization. Returns -min_area."""
    points = x.reshape(N, 2)
    
    xs = points[:, 0]
    ys = points[:, 1]
    
    # Check bounds and triangle constraints
    # 1. y >= 0
    if np.any(ys < 0):
        return 1e6
    # 2. x >= 0
    if np.any(xs < 0):
        return 1e6
    # 3. x <= 1
    if np.any(xs > 1):
        return 1e6
    # 4. y <= sqrt(3) * x
    if np.any(ys > SQRT3 * xs + 1e-9):
        return 1e6
    # 5. y <= -sqrt(3) * (x - 1)
    if np.any(ys > -SQRT3 * (xs - 1) + 1e-9):
        return 1e6
        
    area = _min_area(points)
    return -area

def construct():
    start_time = time.time()
    
    # Bounds for DE: x in [0, 1], y in [0, H]
    bounds = []
    for _ in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, H))
        
    best_x = None
    best_val = 1e9
    
    # Phase 1: Global Search with Differential Evolution
    try:
        res = differential_evolution(
            _objective, 
            bounds, 
            seed=42, 
            maxiter=1000, 
            popsize=20, 
            tol=1e-6,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=False
        )
        best_x = res.x
        best_val = res.fun
    except Exception:
        pass

    # Phase 2: Local Refinement
    elapsed = time.time() - start_time
    if best_x is not None and elapsed < 100:
        try:
            res_local = minimize(
                _objective, 
                best_x, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-9}
            )
            if res_local.fun < best_val:
                best_x = res_local.x
                best_val = res_local.fun
        except Exception:
            pass

    # Phase 3: If time remains, try another DE run with different seed?
    # Or just return the best found.
    # Given the speed, we can afford another run if the first was fast.
    # But N=11 is small, one good DE run should suffice.
    
    if best_x is None:
        # Fallback: Random
        rng = np.random.default_rng(7)
        points = []
        while len(points) < N:
            u, v = rng.random(), rng.random()
            if u + v < 1.0:
                x = v + 0.5 * (1 - u - v)
                y = H * (1 - u - v)
                points.append([x, y])
        return np.array(points)

    return best_x.reshape(N, 2)
