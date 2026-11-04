"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

N = 13
TRI_INDICES = list(combinations(range(N), 3))
NUM_TRIS = len(TRI_INDICES)  # 286

# Precompute index arrays for vectorized area computation
_I = np.array([t[0] for t in TRI_INDICES])
_J = np.array([t[1] for t in TRI_INDICES])
_K = np.array([t[2] for t in TRI_INDICES])

def _tri_areas(pts):
    """Vectorized triangle areas for all 286 triples."""
    pi = pts[_I]
    pj = pts[_J]
    pk = pts[_K]
    v1 = pj - pi
    v2 = pk - pi
    return 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])

def _hull_area(pts):
    try:
        return ConvexHull(pts).volume
    except Exception:
        return 0.0

def _score(pts):
    """Exact score: min triangle area / convex hull area."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2) or np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
        return 0.0
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 0.0
    return float(np.min(_tri_areas(pts)) / ha)

def _soft_min(areas, tau):
    """
    Compute smooth minimum using log-sum-exp.
    """
    max_area = np.max(areas)
    log_sum_exp = np.log(np.sum(np.exp(-(areas - max_area) / tau)))
    return max_area - tau * log_sum_exp

def _neg_soft_min_ratio(pts, temp):
    """
    Smooth surrogate to maximize: -soft_min(areas) / hull_area.
    """
    pts = np.asarray(pts, dtype=np.float64)
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 1e8
    a = _tri_areas(pts)
    a = np.maximum(a, 1e-15)
    s = -a / temp
    s = s - np.max(s)
    logsum = np.log(np.sum(np.exp(s)))
    soft_min = -logsum * temp
    return soft_min / ha

def _decode_fourier(params):
    """
    Decode Fourier parameters into 2D points.
    params: [cx, cy, a1, b1, a2, b2, a3, b3, ...]
    x_i = cx + sum a_k cos(2*pi*k*i/N)
    y_i = cy + sum b_k sin(2*pi*k*i/N)
    """
    cx, cy = params[0], params[1]
    i = np.arange(N)
    x = cx
    y = cy
    for k in range(1, 4):  # Use first 3 harmonics
        a_k = params[2 + 2*(k-1)]
        b_k = params[3 + 2*(k-1)]
        x += a_k * np.cos(2 * np.pi * k * i / N)
        y += b_k * np.sin(2 * np.pi * k * i / N)
    return np.column_stack([x, y])

def _objective_fourier(params, tau=0.01):
    """
    Objective function for Fourier parameters.
    """
    pts = _decode_fourier(params)
    ha = _hull_area(pts)
    if ha < 1e-10:
        return 1e10
    
    a = _tri_areas(pts)
    if np.any(a == 0):
        return 1e10
        
    smooth_min = _soft_min(a, tau)
    score = smooth_min / ha
    return -score

def _refine_cartesian(pts, time_budget):
    """
    Refine a Cartesian point set using L-BFGS on a smooth surrogate.
    """
    t0 = time.time()
    best = pts.copy()
    best_s = _score(pts)
    
    # Use a moderately small temperature for the soft-min to guide L-BFGS
    temp = 0.005
    
    def obj(x):
        p = x.reshape(N, 2)
        return _neg_soft_min_ratio(p, temp)
    
    try:
        res = minimize(obj, best.flatten(), method='L-BFGS-B',
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        cand = res.x.reshape(N, 2)
        s = _score(cand)
        if s > best_s:
            best_s = s
            best = cand.copy()
    except Exception:
        pass
        
    # Second pass with tighter temperature if time permits
    if time.time() - t0 < time_budget - 1:
        temp2 = 0.001
        def obj2(x):
            p = x.reshape(N, 2)
            return _neg_soft_min_ratio(p, temp2)
        try:
            res = minimize(obj2, best.flatten(), method='L-BFGS-B',
                           options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best = cand.copy()
        except Exception:
            pass
            
    return best, best_s

def construct():
    """Main entry: Fourier Spectral Optimization + Local Polish."""
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin for 150s limit
    
    # Initial points: Regular polygon
    th = np.linspace(0, 2 * np.pi, N, endpoint=False)
    r_init = np.ones(N)
    init_pts = np.column_stack([r_init * np.cos(th), r_init * np.sin(th)])
    
    best_pts = init_pts.copy()
    best_score = _score(init_pts)
    
    # 1. Global Search in Fourier Space
    # Parameters: cx, cy, a1, b1, a2, b2, a3, b3
    # Bounds: Center [-1, 1], Amplitudes [-1, 1]
    bounds = [(-1, 1)] * 8
    
    # Multi-start Differential Evolution in Fourier space
    # DE is good for global exploration of this low-dimensional space
    try:
        # We run DE for a fixed number of iterations or time
        # Using a wrapper to enforce time limit
        start_time = time.time()
        
        # Define a wrapper for DE that checks time
        def de_obj(params):
            if time.time() > deadline - 5:
                return 1e10
            return _objective_fourier(params, tau=0.02)
            
        # Initial population: Regular polygon corresponds to a1=1, b1=0, others 0
        # cx=0, cy=0, a1=1, b1=0, a2=0, b2=0, a3=0, b3=0
        x0 = np.array([0, 0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Run DE
        res_de = differential_evolution(
            de_obj, 
            bounds, 
            seed=42, 
            maxiter=50, 
            popsize=20,
            tol=1e-6,
            polish=False  # We will do our own polish
        )
        
        if res_de.fun < -best_score:
            x_opt = res_de.x
            pts_opt = _decode_fourier(x_opt)
            score_opt = _score(pts_opt)
            if score_opt > best_score:
                best_score = score_opt
                best_pts = pts_opt
                
    except Exception:
        pass
    
    # 2. Local Refinement in Cartesian Space
    remaining = deadline - time.time() - 2
    if remaining > 5:
        try:
            pts, s = _refine_cartesian(best_pts, remaining)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
        except Exception:
            pass
            
    # 3. Final Check and Return
    return best_pts
