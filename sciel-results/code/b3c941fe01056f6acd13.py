"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution
import sys

def _all_triangle_areas(pts):
    """Compute areas of all C(n,3) triangles formed by n points."""
    n = len(pts)
    areas = np.empty((n*(n-1)*(n-2))//6)
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = pts[i]
                b = pts[j]
                c = pts[k]
                areas[idx] = 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1]))
                idx += 1
    return areas

def _convex_hull_area(pts):
    """Compute area of convex hull."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    areas = _all_triangle_areas(pts)
    min_area = np.min(areas)
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-12:
        return 0.0
    return min_area / hull_area

def _decode_params(params):
    """
    Decode parameters to Cartesian points.
    params: array of shape (2*N,) = [log_r_0, ..., log_r_{N-1}, theta_0, ..., theta_{N-1}]
    log_r: log of radius (bounded)
    theta: angles, which we sort to enforce convex ordering
    """
    n = N
    log_r = params[:n]
    theta = params[n:]
    
    # Sort by angle to enforce convex ordering
    sort_idx = np.argsort(theta)
    theta_sorted = theta[sort_idx]
    log_r_sorted = log_r[sort_idx]
    
    # Convert to Cartesian
    r = np.exp(log_r_sorted)
    x = r * np.cos(theta_sorted)
    y = r * np.sin(theta_sorted)
    
    return np.column_stack([x, y])

def _objective(params):
    """
    Objective for differential evolution.
    Minimize negative score = -min_triangle_area / hull_area
    """
    try:
        pts = _decode_params(params)
        
        # Check for degenerate configurations
        if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
            return 1e6
        
        # Check minimum distance between points
        min_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < min_dist:
                    min_dist = d
        if min_dist < 1e-6:
            return 1e6
        
        hull_area = _convex_hull_area(pts)
        if hull_area < 1e-10:
            return 1e6
        
        areas = _all_triangle_areas(pts)
        min_area = np.min(areas)
        
        score = min_area / hull_area
        return -score
        
    except Exception:
        return 1e6

def _local_refine(pts, deadline):
    """
    Local refinement using L-BFGS-B on Cartesian coordinates
    with a smooth approximation of the min triangle area.
    """
    start_time = time.time()
    
    def _smooth_min_objective(x, beta):
        pts = x.reshape(-1, 2)
        
        # Compute all triangle areas
        areas = []
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    a = pts[i]
                    b = pts[j]
                    c = pts[k]
                    areas.append(0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1])))
        areas = np.array(areas)
        
        hull_area = _convex_hull_area(pts)
        if hull_area < 1e-10:
            return 1e6
        
        # Soft min: -lse(-beta*areas)/(beta*hull_area) approximates -min(areas)/hull_area
        beta_safe = max(beta, 1.0)
        log_terms = -beta_safe * areas
        max_log = np.max(log_terms)
        lse = max_log + np.log(np.sum(np.exp(log_terms - max_log)))
        obj = -lse / (beta_safe * hull_area)
        
        return obj
    
    best_pts = pts.copy()
    best_score = _score(pts)
    
    for beta in [20, 50, 100, 200]:
        if time.time() > deadline:
            break
        try:
            result = minimize(
                _smooth_min_objective,
                best_pts.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result.fun < 1e5:
                pts_opt = result.x.reshape(-1, 2)
                s = _score(pts_opt)
                if s > best_score:
                    best_score = s
                    best_pts = pts_opt.copy()
        except Exception:
            pass
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    # Parameter bounds for differential evolution
    # log_r: radius from 0.1 to 2.0 (log: -2.3 to 0.7)
    # theta: 0 to 2*pi
    bounds = [(-2.3, 0.7)] * N + [(0, 2*np.pi)] * N
    
    # Initial population: mix of structured and random
    rng = np.random.default_rng(42)
    
    # Create initial points for DE
    # Use a combination of regular polygon, concentric rings, and random
    init_pop = []
    
    # 1. Regular 13-gon
    angles_reg = np.linspace(0, 2*np.pi, N, endpoint=False)
    r_reg = np.ones(N)
    params_reg = np.concatenate([np.log(r_reg), angles_reg])
    init_pop.append(params_reg)
    
    # 2. Concentric rings (center + 2 rings)
    # Center
    params_rings = np.zeros(2*N)
    params_rings[0] = np.log(0.01)  # near center
    params_rings[N] = 0.0  # angle 0
    # Ring 1: 6 points
    for i in range(1, 7):
        params_rings[i] = np.log(0.5)
        params_rings[N+i] = i * 2*np.pi/6
    # Ring 2: 6 points
    for i in range(7, 13):
        params_rings[i] = np.log(1.0)
        params_rings[N+i] = (i-6) * 2*np.pi/6 + np.pi/6
    init_pop.append(params_rings)
    
    # 3. Random points
    for _ in range(8):
        log_r = rng.uniform(-2.0, 0.5, N)
        theta = rng.uniform(0, 2*np.pi, N)
        init_pop.append(np.concatenate([log_r, theta]))
    
    # Run differential evolution
    best_result = None
    best_score = -np.inf
    
    if time.time() < deadline - 30:  # Need at least 30s for local refinement
        try:
            de_result = differential_evolution(
                _objective,
                bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                init='sobol',
                updating='deferred',
                polish=True,
                x0=init_pop,
                constraints=[],
                workers=1
            )
            
            if de_result.success or de_result.fun < 1e5:
                best_params = de_result.x
                best_pts = _decode_params(best_params)
                best_score = _score(best_pts)
                
        except Exception as e:
            # If DE fails, use best from initial population
            for pop in init_pop:
                if time.time() > deadline - 10:
                    break
                try:
                    pts = _decode_params(pop)
                    s = _score(pts)
                    if s > best_score:
                        best_score = s
                        best_pts = pts.copy()
                except Exception:
                    pass
    
    # Fallback: evaluate initial candidates if DE didn't run
    if best_score == -np.inf:
        for pop in init_pop:
            if time.time() > deadline - 10:
                break
            try:
                pts = _decode_params(pop)
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_pts = pts.copy()
            except Exception:
                pass
    
    # Local refinement
    if best_score > -np.inf and time.time() < deadline - 5:
        best_pts, best_score = _local_refine(best_pts, deadline)
    
    # Final fallback
    if best_score == -np.inf:
        # Regular polygon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score = _score(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
