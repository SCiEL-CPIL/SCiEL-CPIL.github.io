"""
Rewritten program for Heilbronn triangles problem.
Strategy: Multi-start L-BFGS-B optimization using a smooth soft-min surrogate.
"""

import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize

N = 11
H = np.sqrt(3) / 2.0
# Precompute all C(11,3) = 165 index triples
_IDX = np.array(list(combinations(range(N), 3)))

def _min_triangle_area_vec(points):
    """
    Compute the minimum area of any triangle formed by three points.
    points: (N, 2) array
    """
    i_idx = _IDX[:, 0]
    j_idx = _IDX[:, 1]
    k_idx = _IDX[:, 2]
    
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    
    v1 = pj - pi
    v2 = pk - pi
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _soft_min_area(points, t=100.0):
    """
    Compute a smooth approximation of the minimum triangle area.
    Uses log-sum-exp trick for stability.
    """
    i_idx = _IDX[:, 0]
    j_idx = _IDX[:, 1]
    k_idx = _IDX[:, 2]
    
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    
    v1 = pj - pi
    v2 = pk - pi
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft min: -1/t * log(sum(exp(-t * areas)))
    # To avoid overflow/underflow, shift by max area
    max_area = np.max(areas)
    if max_area < 1e-15:
        return 0.0
    exp_term = np.exp(-t * (areas - max_area))
    log_sum = np.log(np.sum(exp_term))
    soft_min = max_area - (1.0 / t) * log_sum
    
    return soft_min

def _penalty(points):
    """
    Calculate penalty for points outside the triangle.
    Triangle: (0,0), (1,0), (0.5, H)
    Inequalities:
    1. y >= 0
    2. y <= H * (2x) / 1? No.
       Line from (0,0) to (0.5, H): y = 2H x = sqrt(3) x
       Line from (1,0) to (0.5, H): y = -2H (x-1) = -sqrt(3)(x-1) = sqrt(3)(1-x)
    """
    x = points[:, 0]
    y = points[:, 1]
    
    # y >= 0
    d1 = np.maximum(0, -y)
    # y <= sqrt(3) * x
    d2 = np.maximum(0, y - np.sqrt(3) * x)
    # y <= sqrt(3) * (1 - x)
    d3 = np.maximum(0, y - np.sqrt(3) * (1 - x))
    
    return np.sum(d1**2) + np.sum(d2**2) + np.sum(d3**2)

def _objective_smooth(params, t=100.0):
    """
    Objective function for L-BFGS-B.
    Minimizes -soft_min_area + penalty.
    """
    points = params.reshape(-1, 2)
    area = _soft_min_area(points, t)
    pen = _penalty(points)
    return -area + 1000.0 * pen

def _objective_hard(params):
    """
    Objective function for final polishing (Nelder-Mead).
    Minimizes -min_area + penalty.
    """
    points = params.reshape(-1, 2)
    area = _min_triangle_area_vec(points)
    pen = _penalty(points)
    return -area + 1000.0 * pen

def _clamp_to_triangle(p):
    """
    Project a point to the interior of the triangle.
    """
    x, y = p
    eps = 1e-10
    
    # y bounds
    y = np.clip(y, eps, H - eps)
    
    # x bounds given y
    # y <= sqrt(3) x  => x >= y/sqrt(3)
    # y <= sqrt(3) (1-x) => x <= 1 - y/sqrt(3)
    x_min = y / np.sqrt(3)
    x_max = 1 - y / np.sqrt(3)
    
    x = np.clip(x, x_min + eps, x_max - eps)
    
    return np.array([x, y])

def _clamp_all(points):
    """
    Clamp all points to the triangle.
    """
    return np.array([_clamp_to_triangle(p) for p in points])

def _generate_initial_configs():
    """
    Generate diverse initial configurations.
    """
    configs = []
    
    # 1. Symmetric Configuration
    # Vertices, Centroid, Midpoints, etc.
    sym = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, H],
        [0.5, H/3],       # Centroid
        [0.25, H/6],
        [0.75, H/6],
        [0.5, 2*H/3],
        [0.125, H/12],
        [0.875, H/12],
        [0.375, H/4],
        [0.625, H/4]
    ])
    sym = _clamp_all(sym)
    configs.append(sym)
    
    # 2. Grid-like Configuration
    # Distribute points roughly evenly
    grid = []
    # 3x3 grid + 2 extra? Or just random grid sampling
    for i in range(3):
        for j in range(3):
            # Map i,j to triangle coords
            # Use barycentric-like distribution
            u = (i + 0.5) / 3.0
            v = (j + 0.5) / 3.0
            # Ensure u+v <= 1
            if u + v <= 1.0:
                # Convert barycentric (1-u-v, u, v) to cartesian
                # P = (1-u-v)*A + u*B + v*C
                # A=(0,0), B=(1,0), C=(0.5, H)
                x = u * 1.0 + v * 0.5
                y = v * H
                grid.append([x, y])
    # We need 11 points. The 3x3 grid in a triangle has 6 points (1,2,3).
    # Let's just use a finer grid or random points for the rest.
    # Actually, let's just generate 11 random points and one structured set.
    
    # 3. Random Configurations
    rng = np.random.default_rng(42)
    for i in range(20):
        pts = rng.random((N, 2))
        pts[:, 1] *= H
        pts = _clamp_all(pts)
        configs.append(pts)
        
    # 4. "Voronoi-like" heuristic start
    # Start with vertices and centroid, then push others out
    base = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, H],
        [0.5, H/3]
    ])
    # Add points near midpoints
    for i in range(7):
        angle = 2 * np.pi * i / 7.0
        r = 0.2
        x = 0.5 + r * np.cos(angle)
        y = H/3 + r * np.sin(angle)
        base = np.vstack([base, [x, y]])
    base = _clamp_all(base)
    configs.append(base)
    
    return configs

def construct():
    """
    Place 11 points on or inside a unit-area equilateral triangle to maximize
    the smallest area of a triangle formed by three points.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    configs = _generate_initial_configs()
    
    best_points = None
    best_score = -1.0
    
    # Phase 1: Multi-start L-BFGS-B with smooth objective
    # We use a moderately high t for the soft-min to make it sharp enough.
    # We might try two passes: one with lower t for basin attraction, one with higher t for precision.
    
    for i, config in enumerate(configs):
        if time.time() > deadline - 5.0:
            break
            
        try:
            # Optimization 1: Medium t
            res1 = minimize(
                _objective_smooth,
                config.ravel(),
                args=(50.0,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            
            if res1.success or res1.fun < _objective_smooth(config.ravel(), 50.0):
                pts1 = _clamp_all(res1.x.reshape(-1, 2))
                score1 = _min_triangle_area_vec(pts1)
                
                if score1 > best_score:
                    best_score = score1
                    best_points = pts1.copy()
                    
                # Optimization 2: High t for refinement
                if time.time() < deadline - 2.0:
                    res2 = minimize(
                        _objective_smooth,
                        pts1.ravel(),
                        args=(200.0,),
                        method='L-BFGS-B',
                        options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                    )
                    
                    if res2.success or res2.fun < _objective_smooth(pts1.ravel(), 200.0):
                        pts2 = _clamp_all(res2.x.reshape(-1, 2))
                        score2 = _min_triangle_area_vec(pts2)
                        
                        if score2 > best_score:
                            best_score = score2
                            best_points = pts2.copy()
                            
        except Exception:
            continue

    # Phase 2: Final Polish with Nelder-Mead on the hard objective
    if best_points is not None and time.time() < deadline - 1.0:
        try:
            # Nelder-Mead is derivative-free and can handle non-smooth functions
            # Use the best points as the starting simplex center
            # We need to provide a good initial simplex or let scipy handle it
            # scipy.optimize.minimize with Nelder-Mead takes x0 as a point, 
            # and internally generates a simplex.
            
            res_nm = minimize(
                _objective_hard,
                best_points.ravel(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if res_nm.success or res_nm.fun < _objective_hard(best_points.ravel()):
                pts_nm = _clamp_all(res_nm.x.reshape(-1, 2))
                score_nm = _min_triangle_area_vec(pts_nm)
                
                if score_nm > best_score:
                    best_score = score_nm
                    best_points = pts_nm.copy()
                    
        except Exception:
            pass

    # Fallback if something went wrong
    if best_points is None:
        best_points = configs[0]
        
    return best_points
