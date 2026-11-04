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
    We shift by the minimum to ensure stability.
    """
    min_r = np.min(ratios)
    # shifted values are <= 0
    shifted = -tau * (ratios - min_r)
    # Clip to avoid underflow/overflow issues in exp, though shifted <= 0 is safe for exp
    # exp(shifted) is in (0, 1]
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
    
    # Add a small penalty if the hull area is very small to prevent degenerate scaling
    # Although ratios are normalized, extreme scaling can cause numerical issues in gradient
    penalty = 0.0
    if hull_area < 0.1:
        penalty = 10.0 * (0.1 - hull_area)
        
    return -approx_min + penalty

def _generate_initial_params(rng, method='polygon'):
    """
    Generate initial parameters for the 22 free coordinates.
    """
    if method == 'polygon':
        # Regular 13-gon with random perturbation
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        # Small angular perturbation
        angles += rng.uniform(-0.1, 0.1, N)
        # Small radial perturbation
        radii = 1.0 + rng.normal(0, 0.05, N)
        
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
    
    else:
        return rng.uniform(-1, 1, 22)

def _optimize_single_start(start_params, tau_schedule, time_limit, start_time):
    """
    Run the optimization for a single start point with a given tau schedule.
    """
    current_params = start_params.copy()
    
    for tau in tau_schedule:
        if (time.time() - start_time) > time_limit - 2.0:
            break
        
        try:
            res = minimize(
                _objective,
                current_params,
                args=(tau,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            # Check if the result is better than the current best
            # We compare the smooth objective, but ultimately care about the true score
            # However, for the intermediate steps, smooth objective is a good proxy
            if res.fun < _objective(current_params, tau):
                current_params = res.x.copy()
                
        except Exception:
            pass
            
    return current_params

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(12345)
    
    best_params = None
    best_score = -np.inf
    
    # Phase 1: Generate diverse starting points
    # Focus heavily on the regular polygon prior as it's the strongest
    num_starts = 8
    starts = []
    
    # 4 Polygon starts with different perturbations
    for i in range(4):
        p = _generate_initial_params(rng, 'polygon')
        starts.append(p)
        
    # 2 Halton starts
    for i in range(2):
        p = _generate_initial_params(rng, 'halton')
        starts.append(p)
        
    # 2 Random starts (just in case)
    for i in range(2):
        p = _generate_initial_params(rng, 'random')
        starts.append(p)
    
    # Phase 2: Annealed Optimization
    # Start with low tau (smooth) to find basin, increase tau for precision
    # The transition from smooth to exact is critical
    tau_schedule = [10, 50, 200, 1000, 5000]
    
    for i in range(len(starts)):
        if (time.time() - start_time) > time_limit - 10.0:
            break
            
        # Run optimization for this start
        optimized_params = _optimize_single_start(starts[i], tau_schedule, time_limit, start_time)
        
        # Evaluate true score
        pts = np.empty((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = optimized_params.reshape(N - 2, 2)
        
        score = _compute_min_ratio(pts)
        
        if score > best_score:
            best_score = score
            best_params = optimized_params.copy()
            
    # Phase 3: Refinement of the best solution
    # If we have time, do a final high-precision polish on the best candidate
    if best_params is not None and (time.time() - start_time) < time_limit - 5.0:
        # Use a very high tau for the final polish to approximate the exact min closely
        # This acts as a "zoom in" on the local minimum of the min function
        final_tau_schedule = [2000, 10000, 50000]
        
        current_best = best_params.copy()
        for tau in final_tau_schedule:
            if (time.time() - start_time) > time_limit - 2.0:
                break
            try:
                res = minimize(
                    _objective,
                    current_best,
                    args=(tau,),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                if res.fun < _objective(current_best, tau):
                    current_best = res.x.copy()
            except Exception:
                pass
        
        # Final check
        pts = np.empty((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = current_best.reshape(N - 2, 2)
        final_score = _compute_min_ratio(pts)
        
        if final_score > best_score:
            best_score = final_score
            best_params = current_best.copy()
            
    # Construct final points
    if best_params is None:
        best_params = _generate_initial_params(rng, 'polygon')
    
    final_pts = np.empty((N, 2))
    final_pts[0] = [0.0, 0.0]
    final_pts[1] = [1.0, 0.0]
    final_pts[2:] = best_params.reshape(N - 2, 2)
    
    return final_pts

# EVOLVE-BLOCK-END
