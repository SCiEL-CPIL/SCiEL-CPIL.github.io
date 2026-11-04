"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
import sys
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13
NUM_TRIPLES = N * (N - 1) * (N - 2) // 6

# Precompute triple indices
_triple_indices = np.array(list(combinations(range(N), 3)))

def _triangle_areas(pts):
    """Compute all triangle areas efficiently."""
    i = _triple_indices[:, 0]
    j = _triple_indices[:, 1]
    k = _triple_indices[:, 2]
    v1 = pts[j] - pts[i]
    v2 = pts[k] - pts[i]
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

# EVOLVE-BLOCK-START

def _compute_hull_area(pts):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        area = hull.volume
        if area < 1e-9:
            return 0.0
        return area
    except Exception:
        return 0.0

def _compute_min_ratio(pts):
    """
    Compute the minimum triangle area divided by the convex hull area.
    Returns the score (higher is better) or 0 if invalid.
    """
    hull_area = _compute_hull_area(pts)
    if hull_area == 0.0:
        return 0.0
    
    areas = _triangle_areas(pts)
    min_area = np.min(areas)
    return min_area / hull_area

def _smooth_min(ratios, tau):
    """
    Smooth approximation of the minimum using log-sum-exp.
    min(x) ≈ -1/tau * log(sum(exp(-tau * x)))
    """
    min_r = np.min(ratios)
    shifted = -tau * (ratios - min_r)
    shifted = np.clip(shifted, -50, 0)
    max_shifted = np.max(shifted)
    log_sum = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    return min_r - (1.0 / tau) * log_sum

def _objective(params, tau):
    """
    Objective function: minimize -smooth_min(ratios).
    """
    pts = np.empty((N, 2))
    pts[0] = [0.0, 0.0]
    pts[1] = [1.0, 0.0]
    pts[2:] = params.reshape(N - 2, 2)
    
    hull_area = _compute_hull_area(pts)
    if hull_area == 0.0:
        return 1e9
    
    areas = _triangle_areas(pts)
    ratios = areas / hull_area
    approx_min = _smooth_min(ratios, tau)
    return -approx_min

def _exact_objective(params):
    """
    Objective function for Nelder-Mead: minimize -min_ratio.
    """
    pts = np.empty((N, 2))
    pts[0] = [0.0, 0.0]
    pts[1] = [1.0, 0.0]
    pts[2:] = params.reshape(N - 2, 2)
    
    score = _compute_min_ratio(pts)
    if score == 0.0:
        return 1e9
    return -score

def _transform_to_fixed(pts):
    """Transform points so p[0]=(0,0) and p[1]=(1,0), return params for p[2:]"""
    pts = pts - pts[0]
    v = pts[1]
    length = np.linalg.norm(v)
    if length < 1e-9:
        return None
    pts = pts / length
    theta = np.arctan2(pts[1, 1], pts[1, 0])
    c, s = np.cos(-theta), np.sin(-theta)
    R = np.array([[c, -s], [s, c]])
    pts = pts @ R.T
    return pts[2:].flatten()

def _generate_initial_params(rng, method='polygon'):
    """
    Generate initial parameters for the 22 free coordinates.
    """
    if method == 'polygon':
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        angles += rng.uniform(-0.2, 0.2, N)
        radii = 1.0 + rng.normal(0, 0.1, N)
        
        pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        params = _transform_to_fixed(pts)
        if params is not None:
            return params
        return rng.uniform(-1, 1, 22)
    
    elif method == 'random':
        return rng.uniform(-1.5, 1.5, 22)
    
    elif method == 'halton':
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r
        
        pts = np.zeros((N, 2))
        for i in range(N):
            pts[i, 0] = halton(i + 1, 2)
            pts[i, 1] = halton(i + 1, 3)
        
        pts += rng.normal(0, 0.05, (N, 2))
        
        params = _transform_to_fixed(pts)
        if params is not None:
            return params
        return rng.uniform(-1, 1, 22)
    
    elif method == 'concentric':
        outer_angles = np.linspace(0, 2*np.pi, 7, endpoint=False) + rng.uniform(-0.1, 0.1, 7)
        inner_angles = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6 + rng.uniform(-0.1, 0.1, 6)
        
        pts = np.vstack([
            np.column_stack((np.cos(outer_angles), np.sin(outer_angles))),
            0.5 * np.column_stack((np.cos(inner_angles), np.sin(inner_angles)))
        ])
        params = _transform_to_fixed(pts)
        if params is not None:
            return params
        return rng.uniform(-1, 1, 22)
    
    else:
        return rng.uniform(-1.5, 1.5, 22)

def construct():
    start_time = time.time()
    time_limit = 142.0  # Conservative safety margin before 150s kill
    
    rng = np.random.default_rng(12345)
    
    best_params = None
    best_score = -np.inf
    
    # Define bounds for the free coordinates (22 variables)
    bounds = [(-2.5, 2.5)] * (2 * (N - 2))
    
    # Phase 1: Quick global search using Differential Evolution
    if (time.time() - start_time) < time_limit - 40.0:
        try:
            tau_global = 30.0
            
            de_res = differential_evolution(
                _objective,
                bounds,
                args=(tau_global,),
                maxiter=60,
                popsize=15,
                tol=1e-3,
                seed=12345,
                polish=False,
                init='sobol'
            )
            
            if de_res.x is not None:
                pts = np.empty((N, 2))
                pts[0] = [0.0, 0.0]
                pts[1] = [1.0, 0.0]
                pts[2:] = de_res.x.reshape(N - 2, 2)
                score = _compute_min_ratio(pts)
                
                if score > best_score:
                    best_score = score
                    best_params = de_res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Generate diverse candidates and polish
    candidates = []
    if best_params is not None:
        candidates.append(best_params.copy())
    
    # Add diverse starting points
    methods = ['polygon', 'halton', 'random', 'concentric', 'polygon', 'halton']
    for method in methods:
        if (time.time() - start_time) > time_limit - 15.0:
            break
        p = _generate_initial_params(rng, method)
        candidates.append(p)
    
    # Polish each candidate with increasing tau
    tau_schedule = [30, 100, 500, 2000, 8000]
    
    for tau in tau_schedule:
        if (time.time() - start_time) > time_limit - 8.0:
            break
        
        for i in range(len(candidates)):
            if (time.time() - start_time) > time_limit - 5.0:
                break
            
            try:
                res = minimize(
                    _objective,
                    candidates[i],
                    args=(tau,),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-11}
                )
                
                if res.fun < _objective(candidates[i], tau):
                    candidates[i] = res.x.copy()
                
                pts = np.empty((N, 2))
                pts[0] = [0.0, 0.0]
                pts[1] = [1.0, 0.0]
                pts[2:] = candidates[i].reshape(N - 2, 2)
                score = _compute_min_ratio(pts)
                
                if score > best_score:
                    best_score = score
                    best_params = candidates[i].copy()
                    
            except Exception:
                pass
    
    # Phase 3: Nelder-Mead polish on the exact objective for top candidates
    if (time.time() - start_time) < time_limit - 10.0:
        # Sort candidates by their current score
        scored_candidates = []
        for c in candidates:
            pts = np.empty((N, 2))
            pts[0] = [0.0, 0.0]
            pts[1] = [1.0, 0.0]
            pts[2:] = c.reshape(N - 2, 2)
            score = _compute_min_ratio(pts)
            scored_candidates.append((score, c))
        
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Try NM on the top 3 unique candidates
        top_candidates = []
        for score, c in scored_candidates:
            is_new = True
            for tc in top_candidates:
                if np.linalg.norm(c - tc) < 1e-3:
                    is_new = False
                    break
            if is_new:
                top_candidates.append(c)
            if len(top_candidates) >= 3:
                break
        
        for c in top_candidates:
            if (time.time() - start_time) > time_limit - 5.0:
                break
            
            try:
                res_nm = minimize(
                    _exact_objective,
                    c,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                
                pts = np.empty((N, 2))
                pts[0] = [0.0, 0.0]
                pts[1] = [1.0, 0.0]
                pts[2:] = res_nm.x.reshape(N - 2, 2)
                score = _compute_min_ratio(pts)
                
                if score > best_score:
                    best_score = score
                    best_params = res_nm.x.copy()
                    
            except Exception:
                pass
    
    # Phase 4: Final high-precision polish with L-BFGS-B on soft-min
    if best_params is not None and (time.time() - start_time) < time_limit - 2.0:
        try:
            res = minimize(
                _objective,
                best_params,
                args=(20000,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            if res.fun < _objective(best_params, 20000):
                best_params = res.x.copy()
                pts = np.empty((N, 2))
                pts[0] = [0.0, 0.0]
                pts[1] = [1.0, 0.0]
                pts[2:] = best_params.reshape(N - 2, 2)
                score = _compute_min_ratio(pts)
                if score > best_score:
                    best_score = score
        except Exception:
            pass
    
    # Construct final points
    if best_params is None:
        best_params = _generate_initial_params(rng, 'polygon')
    
    final_pts = np.empty((N, 2))
    final_pts[0] = [0.0, 0.0]
    final_pts[1] = [1.0, 0.0]
    final_pts[2:] = best_params.reshape(N - 2, 2)
    
    return final_pts

# EVOLVE-BLOCK-END
