"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

DEADLINE = 145.0  # Safety margin before 150s kill

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        if len(pts) < 3:
            return 0.0
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _all_triangle_areas(points):
    """Return array of all triangle areas (N choose 3)."""
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    areas_list = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = (x[j] - x[i]) * dy_ki - dx_ki * (y[j] - y[i])
            areas_list.append(0.5 * np.abs(cross))
    if not areas_list:
        return np.array([0.0])
    return np.concatenate(areas_list)

def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area for N points."""
    areas = _all_triangle_areas(points)
    if len(areas) == 0:
        return 0.0
    return np.min(areas)

def _score(pts):
    """Compute min triangle area / hull area."""
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 0.0
    min_area = _min_triangle_area_vec(pts)
    return min_area / hull_a

def _soft_obj(pts_flat, beta=100.0):
    """
    Soft objective to minimize: hull_area / soft_min(areas).
    Minimizing this is equivalent to maximizing soft_min(areas) / hull_area.
    """
    points = pts_flat.reshape(N, 2)
    if np.any(~np.isfinite(points)):
        return 1e10
    
    areas = _all_triangle_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
    
    # Log-sum-exp trick for soft min
    # soft_min(areas) = -1/beta * log(sum(exp(-beta * areas)))
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        # If z is very negative, exp(z) underflows. 
        # This happens if areas are very large relative to 1/beta.
        # soft_min approx min(areas).
        log_sum = max_z
    else:
        exp_z = np.exp(z - max_z)
        log_sum = max_z + np.log(np.sum(exp_z))
    
    soft_min_val = -log_sum / beta
    if soft_min_val < 1e-12:
        return 1e10
        
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e10
        
    return hull_area / soft_min_val

def _neg_score(pts_flat):
    """Negative exact score for minimization."""
    pts = pts_flat.reshape(-1, 2)
    s = _score(pts)
    return -s

def _initial_regular_polygon(n, radius=1.0):
    """Points on a regular n-gon."""
    angles = 2 * np.pi * np.arange(n) / n
    pts = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    return pts

def _initial_perturbed_polygon(n, rng, noise_scale=0.1):
    """Regular polygon with radial and angular noise."""
    angles = 2 * np.pi * np.arange(n) / n
    radii = 1.0 + rng.normal(0, noise_scale, n)
    angles += rng.normal(0, noise_scale / 10, n)
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return pts

def _initial_random_convex(n, rng):
    """Random points on a circle (convex position)."""
    angles = rng.uniform(0, 2*np.pi, n)
    angles.sort()
    radii = 1.0
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return pts

def _initial_random_interior(n, rng):
    """Random points inside a unit circle."""
    angles = rng.uniform(0, 2*np.pi, n)
    radii = rng.uniform(0.5, 1.0, n)
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return pts

def _optimize_candidate(pts_init, time_budget, beta_start=50, beta_end=500):
    """
    Optimize a single candidate using L-BFGS-B with beta annealing.
    """
    start_time = time.time()
    x0 = pts_init.flatten()
    best_x = x0.copy()
    best_obj = _soft_obj(x0, beta_start)
    
    # Simple annealing: few steps with increasing beta
    # We want to converge to the hard min.
    betas = [beta_start, 100, 200, 400, 800]
    
    for beta in betas:
        if time.time() - start_time > time_budget - 2:
            break
            
        def obj_func(x):
            return _soft_obj(x, beta=beta)
            
        try:
            # L-BFGS-B is fast. Limit iterations to keep time under control.
            res = minimize(obj_func, best_x, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6,
                                  'maxfun': 1000})
            if res.x is not None:
                cand_obj = _soft_obj(res.x, beta=beta)
                if cand_obj < best_obj:
                    best_obj = cand_obj
                    best_x = res.x
        except Exception:
            pass
            
    return best_x.reshape(-1, 2)

def _polish_candidate(pts, time_budget):
    """
    Final polish using Nelder-Mead on the exact objective.
    """
    start_time = time.time()
    x0 = pts.flatten()
    best_x = x0.copy()
    best_score = _score(pts)
    
    # Nelder-Mead is derivative-free and robust for small N.
    # Use adaptive Nelder-Mead.
    try:
        res = minimize(_neg_score, best_x, method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14,
                              'adaptive': True, 'maxfev': 20000})
        if res.x is not None:
            cand = res.x.reshape(-1, 2)
            s = _score(cand)
            if s > best_score:
                best_score = s
                best_x = res.x
    except Exception:
        pass
        
    # Try another polish with different initial simplex size?
    # Or just return best.
    return best_x.reshape(-1, 2)

def construct():
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    # Generate initial candidates
    candidates = []
    
    # 1. Regular polygon
    candidates.append(_initial_regular_polygon(N))
    
    # 2. Perturbed regular polygons
    for noise in [0.05, 0.1, 0.2]:
        candidates.append(_initial_perturbed_polygon(N, rng, noise))
        
    # 3. Random convex
    for _ in range(5):
        candidates.append(_initial_random_convex(N, rng))
        
    # 4. Random interior (might get pushed to boundary)
    for _ in range(3):
        candidates.append(_initial_random_interior(N, rng))
        
    # Evaluate initial scores
    scored_candidates = []
    for cand in candidates:
        s = _score(cand)
        scored_candidates.append((s, cand))
        
    # Sort by score descending
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    best_pts = scored_candidates[0][1]
    best_score = scored_candidates[0][0]
    
    # Phase 1: Global Search
    # Allocate ~100s for this
    global_time_budget = 100.0
    time_left_global = global_time_budget - (time.time() - start_time)
    
    # Process top candidates first
    # We will optimize the top 10 candidates
    num_to_optimize = min(10, len(scored_candidates))
    
    # Time per candidate
    time_per_candidate = time_left_global / num_to_optimize
    
    for i in range(num_to_optimize):
        if time.time() - start_time > DEADLINE - 45: # Leave 45s for polish
            break
            
        s_init, cand = scored_candidates[i]
        
        # Time check
        t_start = time.time()
        t_allowed = min(time_per_candidate, DEADLINE - 45 - t_start)
        
        if t_allowed < 1.0:
            continue
            
        try:
            optimized = _optimize_candidate(cand, t_allowed)
            s_opt = _score(optimized)
            if s_opt > best_score:
                best_score = s_opt
                best_pts = optimized
        except Exception:
            pass
            
    # Phase 2: Polish
    # Allocate remaining time minus safety margin
    polish_time_budget = DEADLINE - 5 - (time.time() - start_time)
    
    if polish_time_budget > 2.0:
        try:
            polished = _polish_candidate(best_pts, polish_time_budget)
            s_polished = _score(polished)
            if s_polished > best_score:
                best_score = s_polished
                best_pts = polished
        except Exception:
            pass
            
    return best_pts
# EVOLVE-BLOCK-END
