"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13
NUM_TRIPLES = N * (N - 1) * (N - 2) // 6

# Precompute triple indices
_triple_indices = np.array(list(combinations(range(N), 3)))
_i_idx = _triple_indices[:, 0]
_j_idx = _triple_indices[:, 1]
_k_idx = _triple_indices[:, 2]

# EVOLVE-BLOCK-START

def _compute_hull_area(pts):
    """Compute the area of the convex hull of the points."""
    try:
        # Check for degeneracy
        if np.allclose(pts[:, 0], pts[0, 0]) or np.allclose(pts[:, 1], pts[0, 1]):
            return 0.0
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
    
    p_i = pts[_i_idx]
    p_j = pts[_j_idx]
    p_k = pts[_k_idx]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    return min_area / hull_area

def _smooth_min(ratios, tau):
    """
    Smooth approximation of the minimum using log-sum-exp.
    min(x) ≈ -1/tau * log(sum(exp(-tau * x)))
    """
    min_r = np.min(ratios)
    shifted = -tau * (ratios - min_r)
    # Clip to avoid overflow (shifted should be <= 0)
    shifted = np.clip(shifted, -100, 0)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return min_r - (1.0 / tau) * log_sum

def _objective(params, tau):
    """
    Objective function: minimize -smooth_min(ratios).
    params: 22 values representing p_3 ... p_13.
    p_1 is fixed at (0,0), p_2 is fixed at (1,0).
    """
    pts = np.empty((N, 2))
    pts[0] = [0.0, 0.0]
    pts[1] = [1.0, 0.0]
    pts[2:] = params.reshape(N - 2, 2)
    
    hull_area = _compute_hull_area(pts)
    if hull_area == 0.0:
        return 1e9
    
    p_i = pts[_i_idx]
    p_j = pts[_j_idx]
    p_k = pts[_k_idx]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    ratios = areas / hull_area
    approx_min = _smooth_min(ratios, tau)
    return -approx_min

def _exact_objective(params):
    """
    Exact objective function for Nelder-Mead polishing.
    Minimizes -score.
    """
    pts = np.empty((N, 2))
    pts[0] = [0.0, 0.0]
    pts[1] = [1.0, 0.0]
    pts[2:] = params.reshape(N - 2, 2)
    score = _compute_min_ratio(pts)
    return -score

def _generate_initial_params(rng, method='polygon'):
    """
    Generate initial parameters for the 22 free coordinates.
    """
    if method == 'polygon':
        # Regular 13-gon with random perturbation
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        angles += rng.uniform(-0.15, 0.15, N)
        radii = 1.0 + rng.normal(0, 0.08, N)
        
        pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        
        # Transform: p[0] -> (0,0), p[1] -> (1,0)
        pts = pts - pts[0]
        v = pts[1]
        length = np.linalg.norm(v)
        if length < 1e-9:
            return rng.uniform(-1, 1, 22)
        pts = pts / length
        theta = np.arctan2(pts[1, 1], pts[1, 0])
        c, s = np.cos(-theta), np.sin(-theta)
        R = np.array([[c, -s], [s, c]])
        pts = pts @ R.T
        return pts[2:].flatten()
    
    elif method == 'halton':
        # Halton sequence for low-discrepancy
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
        
        # Add small random perturbation
        pts += rng.normal(0, 0.05, (N, 2))
        
        # Transform
        pts = pts - pts[0]
        v = pts[1]
        length = np.linalg.norm(v)
        if length < 1e-9:
            return rng.uniform(-1, 1, 22)
        pts = pts / length
        theta = np.arctan2(pts[1, 1], pts[1, 0])
        c, s = np.cos(-theta), np.sin(-theta)
        R = np.array([[c, -s], [s, c]])
        pts = pts @ R.T
        return pts[2:].flatten()
        
    elif method == 'golden':
        # Golden Spiral (Fibonacci lattice)
        # This provides a very uniform distribution in 2D
        n = N
        golden_ratio = (1 + np.sqrt(5)) / 2
        # Indices 0 to n-1
        indices = np.arange(n)
        # Radius proportional to sqrt(index) for uniform area density
        r = np.sqrt(indices / n)
        # Angle
        theta = 2 * np.pi * indices / golden_ratio
        
        pts = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        
        # Add small random perturbation to break symmetry
        pts += rng.normal(0, 0.02, (N, 2))
        
        # Transform
        pts = pts - pts[0]
        v = pts[1]
        length = np.linalg.norm(v)
        if length < 1e-9:
            return rng.uniform(-1, 1, 22)
        pts = pts / length
        theta = np.arctan2(pts[1, 1], pts[1, 0])
        c, s = np.cos(-theta), np.sin(-theta)
        R = np.array([[c, -s], [s, c]])
        pts = pts @ R.T
        return pts[2:].flatten()
        
    else: # random
        return rng.uniform(-1, 1, 22)

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(12345)
    
    best_params = None
    best_score = -np.inf
    
    # Phase 1: Generate diverse starting points
    # We use a mix of methods to ensure coverage of the search space
    num_starts = 15
    starts = []
    # Methods: polygon, halton, golden, random
    # We weight towards structured low-discrepancy sets which are often better for this problem
    start_methods = ['polygon', 'halton', 'golden', 'random',
                     'polygon', 'halton', 'golden', 'random',
                     'polygon', 'halton', 'golden', 'random',
                     'golden', 'golden', 'halton']
    
    for i in range(num_starts):
        if (time.time() - start_time) > time_limit - 10.0:
            break
        method = start_methods[i % len(start_methods)]
        p = _generate_initial_params(rng, method)
        starts.append(p)
    
    # Phase 2: Local optimization with L-BFGS-B using smooth min
    # Use multiple tau values for annealing
    # Start with smooth (low tau) to navigate landscape, end with sharp (high tau) to refine min
    tau_schedule = [20, 50, 150, 400]
    
    candidates = [] # List of (score, params)
    
    for tau in tau_schedule:
        if (time.time() - start_time) > time_limit - 5.0:
            break
        
        for i in range(len(starts)):
            if (time.time() - start_time) > time_limit - 3.0:
                break
            
            try:
                # We restart optimization from the current best position for this start
                # This allows us to refine as tau increases
                res = minimize(
                    _objective,
                    starts[i],
                    args=(tau,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if res.fun < _objective(starts[i], tau):
                    starts[i] = res.x.copy()
                
                # Evaluate true score
                pts = np.empty((N, 2))
                pts[0] = [0.0, 0.0]
                pts[1] = [1.0, 0.0]
                pts[2:] = starts[i].reshape(N - 2, 2)
                score = _compute_min_ratio(pts)
                
                if score > best_score:
                    best_score = score
                    best_params = starts[i].copy()
                
                candidates.append((score, starts[i].copy()))
                    
            except Exception:
                pass
    
    # Phase 3: Select top candidates and polish with Nelder-Mead on exact objective
    # Sort candidates by score
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 5 distinct candidates (or all if fewer)
    top_candidates = []
    seen = set()
    for score, params in candidates:
        # Simple check to avoid duplicates (rounding to 6 decimals)
        key = tuple(np.round(params, 6))
        if key not in seen:
            top_candidates.append((score, params))
            seen.add(key)
        if len(top_candidates) >= 5:
            break
            
    if (time.time() - start_time) < time_limit - 10.0:
        for score, params in top_candidates:
            if (time.time() - start_time) > time_limit - 2.0:
                break
            try:
                # Nelder-Mead on the exact, non-smooth objective
                # maxfev is kept low to prevent timeout, as NM can be slow in high dims
                res = minimize(
                    _exact_objective,
                    params,
                    method='Nelder-Mead',
                    options={'maxfev': 1000, 'xatol': 1e-6, 'fatol': 1e-9}
                )
                
                if res.fun < -score: # If we found a better score
                    polished_params = res.x.copy()
                    pts = np.empty((N, 2))
                    pts[0] = [0.0, 0.0]
                    pts[1] = [1.0, 0.0]
                    pts[2:] = polished_params.reshape(N - 2, 2)
                    new_score = _compute_min_ratio(pts)
                    
                    if new_score > best_score:
                        best_score = new_score
                        best_params = polished_params.copy()
                        
            except Exception:
                pass

    # Construct final points
    if best_params is None:
        # Fallback
        best_params = _generate_initial_params(rng, 'polygon')
    
    final_pts = np.empty((N, 2))
    final_pts[0] = [0.0, 0.0]
    final_pts[1] = [1.0, 0.0]
    final_pts[2:] = best_params.reshape(N - 2, 2)
    
    return final_pts

# EVOLVE-BLOCK-END
