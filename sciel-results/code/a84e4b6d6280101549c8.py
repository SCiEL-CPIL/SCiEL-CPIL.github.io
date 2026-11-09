"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area."""
import numpy as np
import time
import itertools
from scipy.optimize import differential_evolution, minimize, L_BFGS_B

N = 11
H = np.sqrt(3) / 2
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, H])

# Precompute all C(11,3) = 165 triangle index triplets
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
_M = len(_idx_i)  # 165


def _min_area(points):
    """Compute minimum triangle area among all C(N,3) triples. points: (N,2)."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    cross = np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) -
                   (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0]))
    return 0.5 * np.min(cross)


def _soft_min_area(points, beta=30.0):
    """Smooth approximation of min area using log-sum-exp."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    cross = np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) -
                   (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0]))
    areas = 0.5 * cross
    min_val = np.min(areas)
    shifted = areas - min_val
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return min_val - log_sum / beta


def _is_inside(points):
    """Check if all points are inside the equilateral triangle."""
    x = points[:, 0]
    y = points[:, 1]
    return (np.all(y >= -1e-9) and 
            np.all(y <= np.sqrt(3) * x + 1e-9) and 
            np.all(y <= np.sqrt(3) * (1 - x) + 1e-9))


def _params_to_points(params):
    """Convert 22 parameters to 11 points using smooth barycentric mapping.
    Each point uses 2 parameters mapped via smooth functions to barycentric coords."""
    points = np.zeros((N, 2))
    for i in range(N):
        u = params[2 * i]
        v = params[2 * i + 1]
        # Use smooth mapping: apply sigmoid-like transform to keep in simplex
        # Map u,v from [0,1] to barycentric coords (a,b,c) with a+b+c=1, a,b,c>0
        # a = 1 - s(u) - s(v) where s is a smooth function
        # Better: use the mapping that ensures interior
        # a = exp(-u) / (exp(-u) + exp(-v) + 1), b = exp(-v) / (...), c = 1 / (...)
        # This is smooth and always in simplex
        eu = np.exp(-u * 3.0)
        ev = np.exp(-v * 3.0)
        s = eu + ev + 1.0
        a = eu / s  # weight for C (top)
        b = ev / s  # weight for B (right)  
        c = 1.0 / s  # weight for A (left)
        # Point = a*C + b*B + c*A
        points[i, 0] = a * 0.5 + b * 1.0
        points[i, 1] = a * H
    return points


def _objective(params, beta=30.0):
    """Objective function: negative soft-min area (to minimize)."""
    points = _params_to_points(params)
    return -_soft_min_area(points, beta=beta)


def _objective_exact(params):
    """Objective function: negative exact min area (to minimize)."""
    points = _params_to_points(params)
    return -_min_area(points)


def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s kill
    
    best_params = None
    best_score = -np.inf
    
    bounds = [(0, 1)] * (2 * N)
    
    # Phase 1: Differential Evolution with soft objective
    # Use smaller populations for more restarts
    n_restarts = 0
    de_configs = [
        {'maxiter': 60, 'popsize': 20},
        {'maxiter': 50, 'popsize': 15},
        {'maxiter': 40, 'popsize': 12},
        {'maxiter': 35, 'popsize': 10},
        {'maxiter': 30, 'popsize': 8},
    ]
    
    while time.time() < deadline - 20:
        n_restarts += 1
        cfg = de_configs[min(n_restarts - 1, len(de_configs) - 1)]
        
        try:
            result = differential_evolution(
                _objective,
                bounds,
                maxiter=cfg['maxiter'],
                popsize=cfg['popsize'],
                tol=1e-8,
                seed=n_restarts * 12345 + 42,
                polish=False,  # We'll do our own polish
                init='latinhypercube',
                mutation=(0.3, 1.0),
                recombination=0.8,
                update_interval=1,
            )
        except Exception:
            continue
        
        # Evaluate exact objective
        exact_score = -_objective_exact(result.x)
        if exact_score > best_score:
            best_score = exact_score
            best_params = result.x.copy()
        
        # Quick Nelder-Mead polish after each DE
        if time.time() < deadline - 15:
            try:
                res2 = minimize(
                    _objective_exact,
                    best_params,
                    method='Nelder-Mead',
                    options={'maxiter': 1500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                exact_score2 = -res2.fun
                if exact_score2 > best_score:
                    best_score = exact_score2
                    best_params = res2.x.copy()
            except Exception:
                pass
    
    # Phase 2: Final high-precision polish
    if best_params is not None and time.time() < deadline - 5:
        # Try L-BFGS-B with exact objective (no constraints needed due to parameterization)
        try:
            res3 = minimize(
                _objective_exact,
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 50}
            )
            exact_score3 = -res3.fun
            if exact_score3 > best_score:
                best_score = exact_score3
                best_params = res3.x.copy()
        except Exception:
            pass
        
        # Another Nelder-Mead pass with very tight tolerances
        if time.time() < deadline - 2:
            try:
                res4 = minimize(
                    _objective_exact,
                    best_params,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
                )
                exact_score4 = -res4.fun
                if exact_score4 > best_score:
                    best_score = exact_score4
                    best_params = res4.x.copy()
            except Exception:
                pass
    
    if best_params is None:
        # Fallback: symmetric grid placement
        points = np.zeros((N, 2))
        positions = [
            (0.5, H * 0.8),
            (0.25, H * 0.5),
            (0.75, H * 0.5),
            (0.1, H * 0.2),
            (0.5, H * 0.2),
            (0.9, H * 0.2),
            (0.35, H * 0.65),
            (0.65, H * 0.65),
            (0.2, H * 0.35),
            (0.8, H * 0.35),
            (0.5, H * 0.45),
        ]
        for i, (x, y) in enumerate(positions):
            points[i] = [x, y]
        return points
    
    points = _params_to_points(best_params)
    
    # Ensure points are strictly inside triangle (small epsilon margin)
    eps = 1e-10
    for i in range(N):
        x, y = points[i]
        # Clamp to interior
        y = max(eps, min(y, np.sqrt(3) * x - eps, np.sqrt(3) * (1 - x) - eps))
        x = max(eps, min(x, 1 - eps))
        # Re-clamp y after x adjustment
        y = max(eps, min(y, np.sqrt(3) * x - eps, np.sqrt(3) * (1 - x) - eps))
        points[i] = [x, y]
    
    return points
