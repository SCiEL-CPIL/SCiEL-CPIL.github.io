"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
import sys
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 13

# EVOLVE-BLOCK-START

def _compute_min_tri_area_norm(pts):
    """
    Compute the minimum area of any triangle formed by pts,
    normalized by the area of the convex hull of pts.
    """
    pts = np.asarray(pts)
    if pts.shape != (N, 2):
        return -1e9
        
    # Compute Convex Hull Area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area < 1e-9:
            return -1e9
    except Exception:
        return -1e9

    # Compute all triangle areas
    min_area = float('inf')
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                p1 = pts[i]
                p2 = pts[j]
                p3 = pts[k]
                area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + 
                                  p2[0] * (p3[1] - p1[1]) + 
                                  p3[0] * (p1[1] - p2[1]))
                if area < min_area:
                    min_area = area

    return min_area / hull_area

def _polar_dual_to_points(log_dists, angles=None):
    """
    Convert log support distances to points using polar dual representation.
    
    The convex region is defined as intersection of half-planes:
    x * cos(theta_i) + y * sin(theta_i) <= d_i for i = 0, ..., N-1
    
    The vertices are the intersections of consecutive half-planes.
    """
    if angles is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    d = np.exp(log_dists)
    
    # For each vertex i, it's the intersection of half-plane i-1 and half-plane i
    # (with wrapping)
    points = np.zeros((N, 2))
    for i in range(N):
        i1 = (i - 1) % N
        i2 = i
        
        # Half-plane i1: x*cos(a1) + y*sin(a1) = d1
        # Half-plane i2: x*cos(a2) + y*sin(a2) = d2
        a1 = angles[i1]
        a2 = angles[i2]
        d1 = d[i1]
        d2 = d[i2]
        
        # Solve the 2x2 system
        A = np.array([
            [np.cos(a1), np.sin(a1)],
            [np.cos(a2), np.sin(a2)]
        ])
        b = np.array([d1, d2])
        
        try:
            det = np.linalg.det(A)
            if abs(det) < 1e-12:
                # Parallel lines - degenerate case
                points[i] = [0, 0]
            else:
                points[i] = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            points[i] = [0, 0]
    
    return points

def _points_to_polar_dual(pts, angles=None):
    """
    Convert points to log support distances using polar dual.
    
    For each edge of the convex hull, compute the distance from origin.
    """
    if angles is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    pts = np.asarray(pts)
    
    # Compute support distances for each angle
    log_dists = np.zeros(N)
    for i in range(N):
        a = angles[i]
        # Support function: max over all points of x*cos(a) + y*sin(a)
        vals = pts[:, 0] * np.cos(a) + pts[:, 1] * np.sin(a)
        d = np.max(vals)
        # Ensure positive
        d = max(d, 1e-6)
        log_dists[i] = np.log(d)
    
    return log_dists

def _soft_min_triangle_ratio(pts, beta=100.0):
    """
    Compute a smooth approximation of the minimum triangle area ratio.
    """
    pts = np.asarray(pts)
    if pts.shape != (N, 2):
        return 0.0
        
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area < 1e-9:
            return 0.0
    except Exception:
        return 0.0

    # Compute all triangle areas
    areas = []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                p1 = pts[i]
                p2 = pts[j]
                p3 = pts[k]
                area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + 
                                  p2[0] * (p3[1] - p1[1]) + 
                                  p3[0] * (p1[1] - p2[1]))
                areas.append(area)
    
    areas = np.array(areas)
    
    # Soft-min using log-sum-exp
    max_neg = np.max(-beta * areas)
    log_sum = max_neg + np.log(np.sum(np.exp(-beta * areas - max_neg)))
    soft_min_val = -log_sum / beta
    
    return soft_min_val / hull_area

def _objective_polar(log_dists, angles=None, beta=100.0):
    """
    Objective function for polar dual optimization.
    Minimizes -soft_min_triangle_ratio.
    """
    pts = _polar_dual_to_points(log_dists, angles)
    val = _soft_min_triangle_ratio(pts, beta=beta)
    return -val

def _refine_from_polar(log_dists, angles, start_time, time_limit, beta_schedule=None):
    """
    Refine a solution starting from polar dual parameters.
    """
    if beta_schedule is None:
        beta_schedule = [10.0, 50.0, 100.0, 200.0, 500.0]
    
    current_log_dists = log_dists.copy()
    pts = _polar_dual_to_points(current_log_dists, angles)
    best_score = _compute_min_tri_area_norm(pts)
    best_pts = pts.copy()
    
    for beta in beta_schedule:
        if (time.time() - start_time) > time_limit - 5.0:
            break
            
        def objective(params):
            return _objective_polar(params, angles, beta=beta)
        
        try:
            res = minimize(
                objective, 
                current_log_dists, 
                method='L-BFGS-B', 
                options={'maxiter': 2000, 'maxfun': 50000, 'disp': False}
            )
            
            if res.success:
                pts_new = _polar_dual_to_points(res.x, angles)
                score = _compute_min_tri_area_norm(pts_new)
                if score > best_score:
                    best_score = score
                    best_pts = pts_new.copy()
                current_log_dists = res.x.copy()
                
        except Exception:
            pass
            
    return best_pts, best_score

def _differential_evolution_polar(angles, start_time, time_limit, popsize=15, maxiter=50):
    """
    Use differential evolution to optimize polar dual parameters.
    """
    # Define bounds for log distances
    # Regular polygon with unit circumradius has support distance = cos(pi/N)
    reg_support = np.cos(np.pi / N)
    lower = np.log(reg_support * 0.3)
    upper = np.log(reg_support * 3.0)
    bounds = [(lower, upper)] * N
    
    def de_objective(log_dists):
        pts = _polar_dual_to_points(log_dists, angles)
        score = _compute_min_tri_area_norm(pts)
        return -score  # Minimize negative score
    
    try:
        result = differential_evolution(
            de_objective,
            bounds,
            seed=42,
            popsize=popsize,
            maxiter=maxiter,
            tol=1e-8,
            init='halton',
            polish=True,
            update_interval=1,
            callback=None
        )
        
        if result.success:
            pts = _polar_dual_to_points(result.x, angles)
            score = _compute_min_tri_area_norm(pts)
            return pts, score, result.x
        else:
            # Still use the best found
            pts = _polar_dual_to_points(result.x, angles)
            score = _compute_min_tri_area_norm(pts)
            return pts, score, result.x
            
    except Exception as e:
        return None, -1e9, None

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin from 150s kill limit
    
    # Fixed angles for polar dual representation
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    # Initial log distances for regular polygon
    reg_support = np.cos(np.pi / N)
    init_log_dists = np.full(N, np.log(reg_support))
    
    best_pts = None
    best_score = -1e9
    
    # Phase 1: Differential evolution in polar dual space
    if (time.time() - start_time) < time_limit - 30.0:
        pts, score, log_dists = _differential_evolution_polar(
            angles, start_time, time_limit, popsize=15, maxiter=30
        )
        if pts is not None and score > best_score:
            best_score = score
            best_pts = pts.copy()
            best_log_dists = log_dists.copy()
    
    # Phase 2: Multi-start refinement from diverse polar dual initializations
    rng = np.random.default_rng(42)
    
    # Start 1: Regular polygon
    starts = [init_log_dists.copy()]
    
    # Start 2: Perturbed regular polygon
    noise = rng.normal(0, 0.1, size=N)
    starts.append(init_log_dists + noise)
    
    # Start 3-8: Random perturbations
    for _ in range(6):
        noise = rng.normal(0, 0.2, size=N)
        starts.append(init_log_dists + noise)
    
    # Start 9-12: More diverse starts
    for _ in range(4):
        noise = rng.normal(0, 0.3, size=N)
        starts.append(init_log_dists + noise)
    
    # Also add the best from DE if we have it
    if 'best_log_dists' in dir() and best_log_dists is not None:
        starts.append(best_log_dists.copy())
    
    for i, init_ld in enumerate(starts):
        if (time.time() - start_time) > time_limit - 15.0:
            break
            
        pts, score = _refine_from_polar(init_ld, angles, start_time, time_limit)
        
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    # Phase 3: Final high-precision polish
    if best_pts is not None and (time.time() - start_time) < time_limit - 10.0:
        # Convert back to polar dual for final polish
        final_log_dists = _points_to_polar_dual(best_pts, angles)
        
        def objective_final(params):
            return _objective_polar(params, angles, beta=500.0)
        
        try:
            res = minimize(
                objective_final, 
                final_log_dists, 
                method='L-BFGS-B', 
                options={'maxiter': 5000, 'maxfun': 100000, 'disp': False}
            )
            
            if res.success:
                pts_final = _polar_dual_to_points(res.x, angles)
                score_final = _compute_min_tri_area_norm(pts_final)
                if score_final > best_score:
                    best_score = score_final
                    best_pts = pts_final.copy()
                    
        except Exception:
            pass
    
    if best_pts is None:
        # Fallback to regular polygon
        pts = _polar_dual_to_points(init_log_dists, angles)
        return pts
    
    return best_pts

# EVOLVE-BLOCK-END
