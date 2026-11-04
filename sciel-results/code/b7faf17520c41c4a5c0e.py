"""Heilbronn triangles: 11 points in unit equilateral triangle, maximize min triangle area."""
import numpy as np
from scipy.optimize import differential_evolution, minimize, basinhopping
import math
import time

N = 11
H = math.sqrt(3) / 2

def _tri_area_vec(pts):
    """
    Calculate the minimum triangle area for N points.
    Optimized for speed using pre-computed indices.
    """
    min_a = float('inf')
    # Pre-compute indices for combinations of 3
    # This is done once per call, but we can cache it? 
    # For N=11, generating combinations is fast.
    # Let's just loop efficiently.
    for i in range(N-2):
        pi = pts[i]
        for j in range(i+1, N-1):
            pj = pts[j]
            dx1 = pj[0] - pi[0]
            dy1 = pj[1] - pi[1]
            for k in range(j+1, N):
                pk = pts[k]
                dx2 = pk[0] - pi[0]
                dy2 = pk[1] - pi[1]
                area = 0.5 * abs(dx1 * dy2 - dx2 * dy1)
                if area < min_a:
                    min_a = area
                    if min_a < 1e-15:
                        return min_a
    return min_a

def _project(p, m=1e-9):
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

def _objective(flat):
    pts = flat.reshape(N, 2)
    # Project to ensure validity
    pts = np.array([_project(p) for p in pts])
    return -_tri_area_vec(pts)

def construct():
    h = H
    # Symmetric initial guess
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
    best_val = _objective(best_flat)
    
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    # 1. Differential Evolution for global search
    try:
        res_de = differential_evolution(
            _objective, bounds, seed=42, maxiter=50, popsize=15,
            tol=1e-6, init='sobol', polish=True, updating='deferred'
        )
        if res_de.fun < best_val:
            best_val = res_de.fun
            best_flat = res_de.x
    except Exception:
        pass
    
    if time.time() - start_time > time_limit:
        points = best_flat.reshape(N, 2)
        points = np.array([_project(p) for p in points])
        return points

    # 2. Basin Hopping for local refinement
    # Use a fast local minimizer. Nelder-Mead is robust but slow.
    # We'll use Nelder-Mead with limited iterations.
    def local_minimizer(x):
        res = minimize(_objective, x, method='Nelder-Mead',
                       options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-11})
        return res.x, res.fun, {}

    try:
        res_bh = basinhopping(
            _objective, best_flat, 
            minimizer=local_minimizer,
            niter=10,  # Reduced iterations for speed
            T=50.0, 
            stepsize=0.3,
            minimizer_kwargs={'options': {'maxiter': 500}}
        )
        if res_bh.fun < best_val:
            best_val = res_bh.fun
            best_flat = res_bh.x
    except Exception:
        pass

    if time.time() - start_time > time_limit:
        points = best_flat.reshape(N, 2)
        points = np.array([_project(p) for p in points])
        return points

    # 3. Final polish with Nelder-Mead
    try:
        res_nm = minimize(_objective, best_flat, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
        if res_nm.fun < best_val:
            best_val = res_nm.fun
            best_flat = res_nm.x
    except Exception:
        pass

    points = best_flat.reshape(N, 2)
    points = np.array([_project(p) for p in points])
    return points
