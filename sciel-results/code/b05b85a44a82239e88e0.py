"""Heilbronn triangles: 11 points in unit equilateral triangle, maximize min triangle area."""
import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
import math

N = 11
H = math.sqrt(3) / 2

def _tri_area(p1, p2, p3):
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_area(pts):
    min_a = float('inf')
    for i, j, k in combinations(range(len(pts)), 3):
        a = _tri_area(pts[i], pts[j], pts[k])
        if a < min_a:
            min_a = a
    return min_a

def _inside(p, m=1e-9):
    x, y = p
    if y < -m or x < -m or x > 1 + m:
        return False
    if y > math.sqrt(3) * x + m:
        return False
    if y > math.sqrt(3) * (1 - x) + m:
        return False
    return True

def _project(p, m=1e-8):
    x, y = p
    if y < m:
        y = m
    min_x = y / math.sqrt(3)
    if x < min_x:
        x = min_x
    max_x = 1 - y / math.sqrt(3)
    if x > max_x:
        x = max_x
    if x < m:
        x = m
    if x > 1 - m:
        x = 1 - m
    return [x, y]

def _neg_min_area(flat):
    pts = flat.reshape(N, 2)
    pts = np.array([_project(p) for p in pts])
    return -_min_area(pts)

def construct():
    h = H
    # Symmetric initial guess: 3 vertices, 3 edge midpoints, 3 inner ring, 1 center, 1 extra
    init = [
        [0.0, 0.0], [1.0, 0.0], [0.5, h],
        [0.5, 0.0], [0.75, h/2], [0.25, h/2],
        [1/3, h/3], [2/3, h/3], [0.5, 2*h/3],
        [0.5, h/3],
        [0.5, h/6]
    ]
    init_flat = np.array(init).flatten()
    
    bounds = [(0, 1), (0, h)] * N
    
    best_flat = init_flat.copy()
    best_val = _neg_min_area(best_flat)
    
    # Differential Evolution
    try:
        res_de = differential_evolution(
            _neg_min_area, bounds, seed=42, maxiter=80, popsize=15,
            tol=1e-6, init='sobol', polish=True, updating='deferred'
        )
        if res_de.fun < best_val:
            best_val = res_de.fun
            best_flat = res_de.x
    except Exception:
        pass
    
    # Polish with Nelder-Mead
    try:
        res_nm = minimize(_neg_min_area, best_flat, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
        if res_nm.fun < best_val:
            best_val = res_nm.fun
            best_flat = res_nm.x
    except Exception:
        pass
    
    # Multiple random restarts with Nelder-Mead
    rng = np.random.RandomState(123)
    for trial in range(8):
        pts = []
        for _ in range(N):
            u, v = rng.random(2)
            if u + v > 1:
                u, v = 1 - u, 1 - v
            p = [u + 0.5 * v, h * v]
            pts.append(_project(p))
        flat0 = np.array(pts).flatten()
        try:
            res = minimize(_neg_min_area, flat0, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_flat = res.x
        except Exception:
            pass
    
    points = best_flat.reshape(N, 2)
    points = np.array([_project(p) for p in points])
    return points
