import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

N = 11
_H = np.sqrt(3) / 2.0
_SQRT3 = np.sqrt(3)

# Precompute triangle indices for vectorization
_combos = list(combinations(range(N), 3))
_idx_i = np.array([c[0] for c in _combos])
_idx_j = np.array([c[1] for c in _combos])
_idx_k = np.array([c[2] for c in _combos])


def _project_to_triangle(pts):
    """
    Project points to stay inside the equilateral triangle 
    with vertices (0,0), (1,0), (0.5, sqrt(3)/2).
    """
    pts = np.array(pts, dtype=float)
    for _ in range(20):
        pts[:, 1] = np.maximum(pts[:, 1], 0)
        pts[:, 0] = np.maximum(pts[:, 0], 0)
        
        mask2 = pts[:, 1] > _SQRT3 * pts[:, 0]
        if np.any(mask2):
            pts[mask2, 1] = _SQRT3 * pts[mask2, 0]
            
        mask3 = pts[:, 1] > _SQRT3 - _SQRT3 * pts[:, 0]
        if np.any(mask3):
            pts[mask3, 1] = _SQRT3 - _SQRT3 * pts[mask3, 0]
            
        pts[:, 1] = np.maximum(pts[:, 1], 0)
        pts[:, 0] = np.clip(pts[:, 0], 0, 1)
        
    return pts


def _min_triangle_area(points):
    """Compute minimum area of all C(11,3) triangles."""
    pi = points[_idx_i]
    pj = points[_idx_j]
    pk = points[_idx_k]
    v1 = pj - pi
    v2 = pk - pi
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return np.min(areas)


def _soft_min(values, beta=100.0):
    """Smooth approximation of the minimum."""
    min_val = np.min(values)
    shifted = values - min_val
    sum_exp = np.sum(np.exp(-beta * shifted))
    return min_val - np.log(sum_exp) / beta


def _penalty(points):
    """Penalty for points outside the triangle."""
    x = points[:, 0]
    y = points[:, 1]
    d1 = np.maximum(0, -y)
    viol2 = np.maximum(0, y - _SQRT3 * x)
    viol3 = np.maximum(0, y - (_SQRT3 - _SQRT3 * x))
    return np.sum(d1**2) + np.sum(viol2**2) + np.sum(viol3**2)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_points = None
    best_score = -np.inf
    
    def objective_lbfgs(flat):
        pts = flat.reshape(N, 2)
        areas = 0.5 * np.abs(
            (pts[_idx_j, 0] - pts[_idx_i, 0]) * (pts[_idx_k, 1] - pts[_idx_i, 1]) -
            (pts[_idx_j, 1] - pts[_idx_i, 1]) * (pts[_idx_k, 0] - pts[_idx_i, 0])
        )
        sm = _soft_min(areas, beta=50.0)
        pen = _penalty(pts)
        return -sm + 10.0 * pen

    def objective_nm(flat):
        pts = _project_to_triangle(flat.reshape(N, 2))
        return -_min_triangle_area(pts)

    rng = np.random.default_rng(42)
    
    starts = []
    for _ in range(10):
        if time.time() > deadline - 10:
            break
        rand_pts = rng.random((N, 2)) * [1.0, _H]
        rand_pts = _project_to_triangle(rand_pts)
        starts.append(rand_pts.flatten())
        
    heuristic = []
    heuristic.extend([[0,0], [1,0], [0.5, _H]])
    heuristic.extend([[0.5, 0], [0.75, _H/2], [0.25, _H/2]])
    heuristic.append([0.5, _H/3])
    heuristic.extend([[0.4, _H/4], [0.6, _H/4], [0.5, _H/2]])
    heuristic_pts = np.array(heuristic)
    starts.append(heuristic_pts.flatten())

    for init_flat in starts:
        if time.time() > deadline - 5:
            break
            
        try:
            res = minimize(objective_lbfgs, init_flat, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            
            res2 = minimize(objective_nm, res.x, method='Nelder-Mead', 
                            options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8})
            
            final_pts = _project_to_triangle(res2.x.reshape(N, 2))
            score = _min_triangle_area(final_pts)
            
            if score > best_score:
                best_score = score
                best_points = final_pts
                
        except Exception:
            continue

    if best_points is None:
        pts = rng.random((N, 2)) * [1.0, _H]
        best_points = _project_to_triangle(pts)
        
    return best_points

# EVOLVE-BLOCK-END
