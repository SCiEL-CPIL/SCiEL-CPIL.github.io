"""
Program to maximize the minimum triangle area of 14 points in a unit area convex hull.
Combines differential evolution in polar coordinates with L-BFGS-B beta-annealing.
"""

import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
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
    
    areas = np.empty((n*(n-1)*(n-2))//6)
    idx = 0
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
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
    min_val = np.min(areas)
    if min_val < 1e-12:
        return 0.0
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

def _polar_to_cartesian(params):
    """
    Convert polar parameters [r0, a0, r1, a1, ...] to Cartesian points.
    Sort by angle to ensure consistent ordering.
    """
    radii = params[0::2]
    angles = params[1::2]
    
    # Sort by angle
    order = np.argsort(angles)
    sorted_angles = angles[order]
    sorted_radii = radii[order]
    
    points = np.column_stack([sorted_radii * np.cos(sorted_angles),
                              sorted_radii * np.sin(sorted_angles)])
    return points

def _cartesian_to_polar(points):
    """
    Convert Cartesian points to polar parameters [r0, a0, r1, a1, ...].
    Sort by angle first.
    """
    x = points[:, 0]
    y = points[:, 1]
    radii = np.sqrt(x**2 + y**2)
    angles = np.arctan2(y, x)
    
    # Sort by angle
    order = np.argsort(angles)
    sorted_radii = radii[order]
    sorted_angles = angles[order]
    
    # Interleave
    params = np.empty(2 * len(points))
    params[0::2] = sorted_radii
    params[1::2] = sorted_angles
    return params

def _negative_score_polar(params):
    """Objective function for differential_evolution in polar coordinates."""
    points = _polar_to_cartesian(params)
    score = _score(points)
    return -score

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
    
    ang2 = 2 * np.pi * np.arange(7) / 7 + np.pi/7
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
    corners = np.array([[1,1],[-1,1],[-1,-1],[1,-1]], dtype=float)
    edge_pts = []
    for i in range(3):
        s = (i+1)/4.0
        edge_pts.append([1, -1+2*s])
        edge_pts.append([-1+2*s, 1])
        edge_pts.append([-1, 1-2*s])
        edge_pts.append([1-2*s, -1])
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

def _polish_with_beta_annealing(x0, start_time, deadline):
    """
    Polish a configuration using L-BFGS-B with beta annealing.
    """
    current_best_x = x0.copy()
    current_best_obj = _objective_exact(x0)
    
    # Define betas for annealing
    betas = [30.0, 60.0, 120.0, 250.0, 500.0, 1000.0]
    
    for beta in betas:
        if time.time() - start_time > deadline - 8:
            break
            
        def obj_func(x, b=beta):
            return _objective(x, b)
            
        try:
            res = minimize(obj_func, current_best_x, method='L-BFGS-B',
                           options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res.success and res.fun < current_best_obj:
                current_best_obj = res.fun
                current_best_x = res.x
        except Exception:
            pass
    
    # Local Polish with Exact Objective (Nelder-Mead)
    if time.time() - start_time < deadline - 5:
        def obj_exact(x):
            return _objective_exact(x)
            
        try:
            res = minimize(obj_exact, current_best_x, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12})
            
            if res.fun < current_best_obj:
                current_best_obj = res.fun
                current_best_x = res.x
        except Exception:
            pass
    
    return current_best_x

def _coordinate_descent_polish(pts, start_time, deadline, num_iters=2, step_sizes=None):
    """
    Coordinate descent polish: move each point slightly to improve min area.
    """
    if step_sizes is None:
        step_sizes = [-0.02, -0.01, 0.01, 0.02]
    
    for cd_iter in range(num_iters):
        if time.time() - start_time > deadline - 3:
            break
        improved = False
        for idx in range(N):
            best_delta = 0.0
            best_pos = pts[idx].copy()
            for dx in step_sizes:
                for dy in step_sizes:
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
    
    return pts

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
            
    # Sort inits by score (best first)
    init_scores = [_score(c) for c in inits]
    sorted_indices = np.argsort(init_scores)[::-1]
    inits = [inits[i] for i in sorted_indices]
    
    # Phase 1: Differential Evolution in polar coordinates
    # This explores the global landscape effectively
    if time.time() - start_time < DEADLINE - 60:
        try:
            # Convert best initial to polar for DE starting point
            # DE will search the full polar space
            bounds = [(0.1, 1.5), (0, 2 * np.pi)] * N
            
            def objective_with_deadline(params):
                if time.time() > DEADLINE - 40:
                    raise TimeoutError()
                return _negative_score_polar(params)
            
            # Use moderate population to stay within time budget
            result = differential_evolution(
                objective_with_deadline,
                bounds,
                maxiter=50,
                popsize=10,
                tol=1e-5,
                seed=42,
                polish=False  # We'll do our own polishing
            )
            
            if result.fun < 0:  # Negative score means positive actual score
                points_de = _polar_to_cartesian(result.x)
                score_de = _score(points_de)
                if score_de > best_score:
                    best_score = score_de
                    best_pts = points_de
        except Exception:
            pass
    
    # Phase 2: Polish top configurations with beta annealing
    # Collect candidates: best from DE + top initial configs
    candidates = [best_pts]
    for init in inits[:5]:
        candidates.append(init)
    
    for i, cand in enumerate(candidates):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        x0 = cand.flatten()
        
        # Beta annealing polish
        polished_x = _polish_with_beta_annealing(x0, start_time, DEADLINE)
        
        # Coordinate descent polish
        pts = polished_x.reshape(N, 2)
        pts = _coordinate_descent_polish(pts, start_time, DEADLINE, 
                                         num_iters=2, 
                                         step_sizes=[-0.02, -0.01, 0.01, 0.02])
        
        # Normalize and score
        cand_pts = _normalize_unit_area(pts)
        s = _score(cand_pts)
        
        if s > best_score:
            best_score = s
            best_pts = cand_pts
    
    # Phase 3: Final high-precision polish on the best solution
    if time.time() - start_time < DEADLINE - 5:
        # More precise beta annealing
        polished_x = _polish_with_beta_annealing(best_pts.flatten(), start_time, DEADLINE)
        pts = polished_x.reshape(N, 2)
        
        # Fine coordinate descent
        pts = _coordinate_descent_polish(pts, start_time, DEADLINE, 
                                         num_iters=3,
                                         step_sizes=[-0.005, -0.002, 0.002, 0.005])
        
        best_pts = _normalize_unit_area(pts)
        best_score = _score(best_pts)
    
    # Final Nelder-Mead polish
    if time.time() - start_time < DEADLINE - 2:
        def obj_final(x):
            return _objective_exact(x)
        
        try:
            res = minimize(obj_final, best_pts.flatten(), method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-14})
            if res.fun < _objective_exact(best_pts.flatten()):
                best_pts = res.x.reshape(N, 2)
        except Exception:
            pass
        
        best_pts = _normalize_unit_area(best_pts)
    
    return best_pts

# The entry point must be construct()
if __name__ == "__main__":
    pts = construct()
    print(_score(pts))
