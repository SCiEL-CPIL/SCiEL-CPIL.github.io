"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area."""
import numpy as np
import time
import itertools
from scipy.optimize import differential_evolution, minimize

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


def _min_area(points):
    """Compute minimum triangle area among all C(N,3) triples. points: (N,2)."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    areas = 0.5 * np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) -
                         (p3[:, 0] - p1[:, 0]) * (p2[:, 1] - p1[:, 1]))
    return float(np.min(areas))


def _is_inside(points):
    """Check if all points are inside or on the boundary of the equilateral triangle."""
    x, y = points[:, 0], points[:, 1]
    # y >= 0
    if np.any(y < -1e-9):
        return False
    # y <= H
    if np.any(y > H + 1e-9):
        return False
    # y <= sqrt(3)*x  (left edge from origin)
    if np.any(y > np.sqrt(3) * x + 1e-9):
        return False
    # y <= -sqrt(3)*(x - 1)  (right edge)
    if np.any(y > -np.sqrt(3) * (x - 1.0) + 1e-9):
        return False
    return True


def _clip_points(points):
    """Clip points to be inside the triangle (project to boundary if outside)."""
    x, y = points[:, 0].copy(), points[:, 1].copy()
    for _ in range(10):
        # Clip to y >= 0
        y = np.maximum(y, 0)
        # Clip to y <= H
        y = np.minimum(y, H)
        # Clip to y <= sqrt(3)*x
        x_min = y / np.sqrt(3)
        x = np.maximum(x, x_min)
        # Clip to y <= -sqrt(3)*(x-1) => x <= 1 - y/sqrt(3)
        x_max = 1.0 - y / np.sqrt(3)
        x = np.minimum(x, x_max)
    return np.column_stack([x, y])


def _params_to_points(params):
    """Convert 2N parameters to N points, clipping to triangle."""
    x = params[0::2]
    y = params[1::2]
    points = np.column_stack([x, y])
    return _clip_points(points)


def _soft_min_area(params, beta=50.0):
    """Smooth approximation of min area using log-sum-exp."""
    points = _params_to_points(params)
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    areas = 0.5 * np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) -
                         (p3[:, 0] - p1[:, 0]) * (p2[:, 1] - p1[:, 1]))
    # Use soft-min: -log(sum(exp(-beta * areas))) / beta
    # For numerical stability
    a_min = np.min(areas)
    log_sum = np.log(np.sum(np.exp(-beta * (areas - a_min))))
    soft_min = a_min - np.log(np.sum(np.exp(-beta * (areas - a_min)))) / beta
    return -soft_min  # We want to minimize negative of soft-min


def _objective(params):
    """Objective for DE: negative of soft-min area."""
    return _soft_min_area(params, beta=50.0)


def _objective_exact(params):
    """Objective for Nelder-Mead: negative of exact min area."""
    points = _params_to_points(params)
    return -_min_area(points)


def _nelder_mead_polish(points, max_iter=500):
    """Polish a configuration using Nelder-Mead on exact min area."""
    params = points.flatten()
    
    def obj(p):
        pts = _params_to_points(p)
        return -_min_area(pts)
    
    res = minimize(obj, params, method='Nelder-Mead',
                   options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12})
    return _params_to_points(res.x)


def _refine_with_local_search(points, n_iter=200):
    """Iteratively improve by moving the point that is in the minimum triangle."""
    best_points = points.copy()
    best_val = _min_area(best_points)
    
    for _ in range(n_iter):
        p1 = best_points[_idx_i]
        p2 = best_points[_idx_j]
        p3 = best_points[_idx_k]
        areas = 0.5 * np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) -
                             (p3[:, 0] - p1[:, 0]) * (p2[:, 1] - p1[:, 1]))
        min_idx = np.argmin(areas)
        i, j, k = _idx_i[min_idx], _idx_j[min_idx], _idx_k[min_idx]
        
        # Try moving each of the three points slightly
        best_move = None
        best_new_val = best_val
        
        for pt_idx in [i, j, k]:
            pt = best_points[pt_idx].copy()
            # Try small perturbations
            for dx in [-0.05, -0.02, 0.0, 0.02, 0.05]:
                for dy in [-0.05, -0.02, 0.0, 0.02, 0.05]:
                    new_pt = pt + np.array([dx, dy])
                    candidate = best_points.copy()
                    candidate[pt_idx] = new_pt
                    candidate = _clip_points(candidate)
                    val = _min_area(candidate)
                    if val > best_new_val:
                        best_new_val = val
                        best_move = candidate.copy()
        
        if best_move is not None:
            best_points = best_move
            best_val = best_new_val
        else:
            break
    
    return best_points, best_val


def construct():
    t0 = time.time()
    deadline = t0 + 110.0  # Safety margin before 150s limit
    
    # Phase 1: Differential Evolution with soft-min surrogate
    bounds = [(0, 1), (0, H)] * N
    result = differential_evolution(
        _objective, bounds, seed=42, maxiter=300, popsize=15,
        tol=1e-8, mutation=(0.5, 1.0), recombination=0.7,
        polish=False, updating='deferred', init='latinhypercube'
    )
    
    best_points = _params_to_points(result.x)
    best_val = _min_area(best_points)
    
    if time.time() > deadline - 30:
        return best_points
    
    # Phase 2: Nelder-Mead polish on exact min area
    polished = _nelder_mead_polish(best_points, max_iter=300)
    polished_val = _min_area(polished)
    if polished_val > best_val:
        best_points = polished
        best_val = polished_val
    
    if time.time() > deadline - 15:
        return best_points
    
    # Phase 3: Local search refinement
    refined, refined_val = _refine_with_local_search(best_points, n_iter=100)
    if refined_val > best_val:
        best_points = refined
        best_val = refined_val
    
    if time.time() > deadline - 5:
        return best_points
    
    # Phase 4: Final Nelder-Mead polish
    final = _nelder_mead_polish(best_points, max_iter=200)
    final_val = _min_area(final)
    if final_val > best_val:
        best_points = final
        best_val = final_val
    
    return best_points
