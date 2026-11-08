"""
Program to maximize the minimum triangle area of 14 points in a unit area convex hull.
"""

import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14
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
    """
    Return array of all triangle areas (N choose 3).
    Vectorized calculation for speed.
    """
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    
    # Pre-allocate or accumulate. N=14, so 364 triangles. Small enough to just loop.
    areas = np.empty((n*(n-1)*(n-2))//6)
    idx = 0
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            # Vectorize over k > j
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = (x[j] - x[i]) * dy_ki - dx_ki * (y[j] - y[i])
            tri_areas = 0.5 * np.abs(cross)
            areas[idx:idx+len(tri_areas)] = tri_areas
            idx += len(tri_areas)
            
    return areas

def _min_triangle_area_vec(points):
    """Compute the minimum triangle area."""
    areas = _all_triangle_areas(points)
    if len(areas) == 0:
        return 0.0
    return np.min(areas)

def _score(pts):
    """Compute min triangle area / hull area."""
    hull_a = _hull_area(pts)
    if hull_a <= 1e-10:
        return 0.0
    min_area = _min_triangle_area_vec(pts)
    return min_area / hull_a

def _soft_min(areas, beta):
    """
    Compute the soft minimum of an array of values.
    soft_min(x) = -1/beta * log(sum(exp(-beta*x)))
    """
    if len(areas) == 0:
        return 0.0
    # Numerical stability
    min_val = np.min(areas)
    if min_val < 1e-12:
        return 0.0
    # Shift to prevent overflow/underflow in exp
    # We want min, so we use -beta * areas.
    # To be stable, we subtract the max of (-beta * areas), which is -beta * min(areas).
    # Let z = -beta * areas. max_z = -beta * min(areas).
    # sum(exp(z - max_z)) = sum(exp(-beta*areas + beta*min(areas)))
    #                      = sum(exp(-beta*(areas - min(areas))))
    # log_sum = max_z + log(sum(...))
    # soft_min = -1/beta * (max_z + log_sum_part)
    
    # Simpler stable form:
    # soft_min = min_val - (1/beta) * log( sum(exp(-beta*(areas - min_val))) )
    diff = areas - min_val
    exp_terms = np.exp(-beta * diff)
    log_sum = np.log(np.sum(exp_terms))
    return min_val - log_sum / beta

def _objective(pts_flat, beta):
    """
    Objective function to minimize.
    We want to maximize min_area / hull_area.
    Equivalent to minimizing hull_area / min_area.
    Using soft_min: minimize hull_area / soft_min(areas, beta).
    """
    points = pts_flat.reshape(-1, 2)
    if np.any(~np.isfinite(points)):
        return 1e10
        
    areas = _all_triangle_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
        
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e10
        
    sm = _soft_min(areas, beta)
    if sm < 1e-12:
        return 1e10
        
    return hull_area / sm

def _objective_exact(pts_flat):
    """
    Objective function for final polishing.
    Minimizes hull_area / min_area.
    """
    points = pts_flat.reshape(-1, 2)
    if np.any(~np.isfinite(points)):
        return 1e10
        
    areas = _all_triangle_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
        
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e10
        
    min_a = np.min(areas)
    return hull_area / min_a

def _normalize_unit_area(pts):
    """Scale points so convex hull area is 1."""
    hull_a = _hull_area(pts)
    if hull_a > 0:
        return pts / np.sqrt(hull_a)
    return pts

def _generate_initial_configs():
    """Generate diverse initial configurations."""
    configs = []
    
    # 1. Regular 14-gon
    angles = 2 * np.pi * np.arange(N) / N
    p_reg = np.column_stack([np.cos(angles), np.sin(angles)])
    configs.append(p_reg)
    
    # 2. Regular 14-gon with noise (multiple seeds)
    rng = np.random.RandomState(42)
    for seed in range(8):
        noise = rng.normal(0, 0.08, size=(N, 2))
        p_noisy = p_reg + noise
        configs.append(p_noisy)
        
    # 3. Two concentric rings (7 and 7)
    ang1 = 2 * np.pi * np.arange(7) / 7
    r1 = 0.55
    p1 = np.column_stack([r1 * np.cos(ang1), r1 * np.sin(ang1)])
    
    ang2 = 2 * np.pi * np.arange(7) / 7 + np.pi/7 # Offset
    r2 = 1.0
    p2 = np.column_stack([r2 * np.cos(ang2), r2 * np.sin(ang2)])
    
    p_rings = np.vstack([p1, p2])
    configs.append(p_rings)
    
    # 4. Three concentric rings (5, 5, 4)
    ang_a = 2 * np.pi * np.arange(5) / 5
    p_a = np.column_stack([0.4 * np.cos(ang_a), 0.4 * np.sin(ang_a)])
    ang_b = 2 * np.pi * np.arange(5) / 5 + np.pi/5
    p_b = np.column_stack([0.7 * np.cos(ang_b), 0.7 * np.sin(ang_b)])
    ang_c = 2 * np.pi * np.arange(4) / 4 + np.pi/4
    p_c = np.column_stack([1.0 * np.cos(ang_c), 1.0 * np.sin(ang_c)])
    p_3rings = np.vstack([p_a, p_b, p_c])
    configs.append(p_3rings)
    
    # 5. Distorted regular polygon (3-fold symmetry)
    mod = 1.0 + 0.12 * np.cos(3 * angles)
    p_mod = np.column_stack([mod * np.cos(angles), mod * np.sin(angles)])
    configs.append(p_mod)
    
    # 6. Elliptical distribution
    t = np.linspace(0, 2*np.pi, N, endpoint=False)
    p_ell = np.column_stack([1.2 * np.cos(t), 0.8 * np.sin(t)])
    configs.append(p_ell)
    
    # 7. Square-ish with corner emphasis
    # 4 corners + 3 edges each side = 16, pick 14
    corners = np.array([[1,1],[-1,1],[-1,-1],[1,-1]], dtype=float)
    edge_pts = []
    for i in range(3):
        s = (i+1)/4.0
        edge_pts.append([1, -1+2*s])  # right edge
        edge_pts.append([-1+2*s, 1])  # top edge
        edge_pts.append([-1, 1-2*s])  # left edge
        edge_pts.append([1-2*s, -1])  # bottom edge
    edge_pts = np.array(edge_pts[:10])
    p_sq = np.vstack([corners, edge_pts])
    configs.append(p_sq)
    
    # 8. Random points in disk
    for seed in [1, 2, 3]:
        rng2 = np.random.RandomState(seed)
        r = rng2.uniform(0, 1, N) ** 0.5
        th = rng2.uniform(0, 2*np.pi, N)
        p_rand = np.column_stack([r * np.cos(th), r * np.sin(th)])
        configs.append(p_rand)
    
    # Normalize all to unit area for consistency in initial scoring
    normalized_configs = [_normalize_unit_area(c) for c in configs]
    return normalized_configs

def construct():
    start_time = time.time()
    
    # Get initial configs
    inits = _generate_initial_configs()
    
    # Evaluate initial scores
    best_score = -np.inf
    best_pts = inits[0]
    
    for init in inits:
        s = _score(init)
        if s > best_score:
            best_score = s
            best_pts = init
            
    # Sort inits by score (best first) to prioritize good starting points
    init_scores = [_score(c) for c in inits]
    sorted_indices = np.argsort(init_scores)[::-1]
    inits = [inits[i] for i in sorted_indices]
    
    # Optimization Loop
    # We will run optimization on the top few initial configurations
    
    # Define betas for annealing - more aggressive schedule
    betas = [30.0, 60.0, 120.0, 250.0, 500.0, 1000.0]
    
    num_inits_to_try = min(8, len(inits))
    
    for i in range(num_inits_to_try):
        if time.time() - start_time > DEADLINE - 15:
            break
            
        x0 = inits[i].flatten()
        current_best_x = x0.copy()
        current_best_obj = _objective_exact(x0)
        
        # Annealing loop
        for beta in betas:
            if time.time() - start_time > DEADLINE - 8:
                break
                
            # Define objective for this beta
            def obj_func(x, b=beta):
                return _objective(x, b)
                
            try:
                # L-BFGS-B is good for smooth functions
                res = minimize(obj_func, current_best_x, method='L-BFGS-B',
                               options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10})
                
                if res.success and res.fun < current_best_obj:
                    current_best_obj = res.fun
                    current_best_x = res.x
            except Exception:
                pass
        
        # Local Polish with Exact Objective (Nelder-Mead)
        if time.time() - start_time < DEADLINE - 5:
            def obj_exact(x):
                return _objective_exact(x)
                
            try:
                res = minimize(obj_exact, current_best_x, method='Nelder-Mead',
                               options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12})
                
                if res.fun < current_best_obj:
                    current_best_obj = res.fun
                    current_best_x = res.x
            except Exception:
                pass
        
        # Coordinate descent polish: move each point slightly to improve min area
        if time.time() - start_time < DEADLINE - 3:
            pts = current_best_x.reshape(N, 2)
            for cd_iter in range(3):
                improved = False
                for idx in range(N):
                    best_delta = 0.0
                    best_pos = pts[idx].copy()
                    # Try small perturbations
                    for dx in [-0.02, -0.01, 0.01, 0.02]:
                        for dy in [-0.02, -0.01, 0.01, 0.02]:
                            trial = pts.copy()
                            trial[idx] = pts[idx] + np.array([dx, dy])
                            s_trial = _score(trial)
                            if s_trial > best_delta:
                                best_delta = s_trial
                                best_pos = trial[idx].copy()
                    if best_delta > 0:
                        pts[idx] = best_pos
                        improved = True
                if not improved:
                    break
            current_best_x = pts.flatten()
        
        # Update global best
        cand_pts = current_best_x.reshape(-1, 2)
        # Normalize to ensure unit area hull for final score consistency
        cand_pts = _normalize_unit_area(cand_pts)
        s = _score(cand_pts)
        
        if s > best_score:
            best_score = s
            best_pts = cand_pts
            
    # Final high-precision polish on the best solution
    if time.time() - start_time < DEADLINE - 2:
        def obj_final(x):
            return _objective_exact(x)
        
        try:
            res = minimize(obj_final, best_pts.flatten(), method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14})
            if res.fun < _objective_exact(best_pts.flatten()):
                best_pts = res.x.reshape(N, 2)
        except Exception:
            pass
        
        # One more coordinate descent pass
        pts = best_pts.copy()
        for cd_iter in range(2):
            improved = False
            for idx in range(N):
                best_delta = 0.0
                best_pos = pts[idx].copy()
                for dx in [-0.005, -0.002, 0.002, 0.005]:
                    for dy in [-0.005, -0.002, 0.002, 0.005]:
                        trial = pts.copy()
                        trial[idx] = pts[idx] + np.array([dx, dy])
                        s_trial = _score(trial)
                        if s_trial > best_delta:
                            best_delta = s_trial
                            best_pos = trial[idx].copy()
                if best_delta > 0:
                    pts[idx] = best_pos
                    improved = True
            if not improved:
                break
        best_pts = pts
    
    return _normalize_unit_area(best_pts)

# The entry point must be construct()
if __name__ == "__main__":
    pts = construct()
    print(_score(pts))
