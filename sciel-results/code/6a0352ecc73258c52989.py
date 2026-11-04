import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from itertools import product

N = 14

# Precompute indices for all triples for fast scoring
def _get_triple_indices(n):
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=int)

_TRIPLE_INDICES = _get_triple_indices(N)

def _score(points):
    """
    Compute the score: min triangle area / convex hull area.
    """
    if len(points) < 3:
        return 0.0
    
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    # Calculate Minimum Triangle Area
    i_idx = _TRIPLE_INDICES[:, 0]
    j_idx = _TRIPLE_INDICES[:, 1]
    k_idx = _TRIPLE_INDICES[:, 2]
    
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    # Cross product in 2D
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def _generate_ring_points(r, theta_offset):
    """
    Generate 14 points: 7 on unit circle, 7 on circle of radius r.
    Outer points are at angles 0, 2pi/7, ..., 12pi/7.
    Inner points are at angles theta_offset, theta_offset + 2pi/7, ..., theta_offset + 12pi/7.
    """
    angles_outer = np.linspace(0, 2*np.pi, 7, endpoint=False)
    angles_inner = theta_offset + np.linspace(0, 2*np.pi, 7, endpoint=False)
    
    outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    inner_pts = r * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    
    pts = np.vstack([outer_pts, inner_pts])
    return pts

def _score_ring_params(params):
    """
    Objective function for parameter optimization.
    Returns negative score for minimization.
    """
    r, theta = params
    if r <= 0 or r >= 1:
        return 1e6 # Penalty for invalid radius
    pts = _generate_ring_points(r, theta)
    s = _score(pts)
    return -s

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_score = 0
    best_points = None
    current_points = None
    current_score = 0
    
    # 1. Check regular 14-gon as a baseline
    angles_reg = np.linspace(0, 2*np.pi, N, endpoint=False)
    init_reg = np.column_stack([np.cos(angles_reg), np.sin(angles_reg)])
    s_reg = _score(init_reg)
    if s_reg > best_score:
        best_score = s_reg
        best_points = init_reg.copy()
        current_points = init_reg.copy()
        current_score = s_reg

    # 2. Grid search over ring parameters (r, theta)
    # r is in (0, 1). theta is in [0, pi/7) due to symmetry.
    # Coarse grid
    r_vals = np.linspace(0.1, 0.95, 20)
    theta_vals = np.linspace(0, np.pi/7, 10, endpoint=False)
    
    best_r = 0.5
    best_theta = np.pi/14
    best_ring_score = 0
    
    for r in r_vals:
        for theta in theta_vals:
            pts = _generate_ring_points(r, theta)
            s = _score(pts)
            if s > best_ring_score:
                best_ring_score = s
                best_r = r
                best_theta = theta
    
    if best_ring_score > best_score:
        best_score = best_ring_score
        best_points = _generate_ring_points(best_r, best_theta).copy()
        current_points = best_points.copy()
        current_score = best_score

    # 3. Local refinement of the best ring configuration
    if time.time() < deadline - 10:
        def objective(x):
            if time.time() > deadline - 2:
                raise TimeoutError()
            return _score_ring_params(x)
            
        try:
            # Start from the best grid point
            initial_guess = [best_r, best_theta]
            
            # Use Nelder-Mead for the 2D parameter space
            res = minimize(objective, initial_guess, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if res.success or res.fun < 0:
                r_opt, theta_opt = res.x
                # Ensure r is within bounds
                if 0 < r_opt < 1:
                    pts_opt = _generate_ring_points(r_opt, theta_opt)
                    s_opt = _score(pts_opt)
                    if s_opt > best_score:
                        best_score = s_opt
                        best_points = pts_opt.copy()
                        current_points = pts_opt.copy()
                        current_score = s_opt
        except TimeoutError:
            pass
        except Exception:
            pass

    # 4. If time permits, try a second local search from a different starting point
    # e.g., near r=0.5, theta=0
    if time.time() < deadline - 5:
        def objective2(x):
            if time.time() > deadline - 1:
                raise TimeoutError()
            return _score_ring_params(x)
            
        try:
            initial_guess2 = [0.5, 0.0]
            res2 = minimize(objective2, initial_guess2, method='Nelder-Mead',
                            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if res2.success or res2.fun < 0:
                r_opt2, theta_opt2 = res2.x
                if 0 < r_opt2 < 1:
                    pts_opt2 = _generate_ring_points(r_opt2, theta_opt2)
                    s_opt2 = _score(pts_opt2)
                    if s_opt2 > best_score:
                        best_score = s_opt2
                        best_points = pts_opt2.copy()
        except TimeoutError:
            pass
        except Exception:
            pass

    if best_points is None:
        return init_reg

    return best_points

# EVOLVE-BLOCK-START
# This block is kept empty as the main logic is in construct() above.
# Any additional helper functions or logic should be placed here if needed,
# but the current strategy relies on the parameter-space search.
# EVOLVE-BLOCK-END
