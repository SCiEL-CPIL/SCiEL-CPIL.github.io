import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
import time
import itertools

N = 14

# Precompute triple indices for N=14
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
])

def _min_triangle_area(points):
    """Return min triangle area (not normalized)."""
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    return 0.5 * np.min(np.abs(cross))

def _calculate_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: (N, 2) array
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _soft_min_score(points, beta=100.0):
    """
    Smooth surrogate for min triangle area / hull area using log-sum-exp.
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    
    areas = 0.5 * np.abs(cross)
    
    # Soft min via log-sum-exp: soft_min = m - log(sum(exp(-beta*(x_i - m))))/beta
    max_area = np.max(areas)
    exp_terms = np.exp(-beta * (areas - max_area))
    sum_exp = np.sum(exp_terms)
    soft_min_area = max_area - np.log(sum_exp) / beta
    
    return soft_min_area / hull_area

def _score_ring_params(n_out, n_in, params):
    """
    Objective function for differential evolution.
    params: [r, theta]
    """
    r, theta = params
    if r < 0 or r > 0.999:
        return 0.0
        
    # Generate outer points
    angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
    x_out = np.cos(angles_out)
    y_out = np.sin(angles_out)
    
    # Generate inner points
    # Rotate inner points by theta
    angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + theta
    x_in = r * np.cos(angles_in)
    y_in = r * np.sin(angles_in)
    
    points = np.vstack([
        np.column_stack((x_out, y_out)),
        np.column_stack((x_in, y_in))
    ])
    
    return _calculate_score(points)

def _soft_score_ring_params(n_out, n_in, params, beta=100.0):
    """
    Soft objective function for gradient-based optimization.
    """
    r, theta = params
    if r < 0 or r > 0.999:
        return 0.0
        
    # Generate outer points
    angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
    x_out = np.cos(angles_out)
    y_out = np.sin(angles_out)
    
    # Generate inner points
    # Rotate inner points by theta
    angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + theta
    x_in = r * np.cos(angles_in)
    y_in = r * np.sin(angles_in)
    
    points = np.vstack([
        np.column_stack((x_out, y_out)),
        np.column_stack((x_in, y_in))
    ])
    
    return _soft_min_score(points, beta)

def _optimize_ring(n_out, n_in, time_limit):
    """
    Optimize a specific ring configuration.
    """
    start_time = time.time()
    
    g = np.gcd(n_out, n_in)
    theta_max = 2 * np.pi / g
    
    bounds = [(0.0, 0.999), (0.0, theta_max)]
    
    best_score = 0.0
    best_params = None
    
    # Phase 1: Global search with Differential Evolution
    try:
        popsize = 12
        maxiter = 40
        
        res = differential_evolution(
            lambda p: -_score_ring_params(n_out, n_in, p),
            bounds,
            seed=42,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-5,
            polish=False
        )
        
        if res.success:
            best_score = -res.fun
            best_params = res.x
    except Exception:
        pass
    
    # Phase 2: Local polish with Soft-Min L-BFGS-B using adaptive beta
    if time.time() - start_time < time_limit * 0.6:
        starts = []
        
        if best_params is not None:
            starts.append(best_params)
            
        # Diverse heuristic starts
        for r_start in [0.3, 0.5, 0.7, 0.9]:
            for theta_start in [0.0, theta_max/3, 2*theta_max/3]:
                starts.append([r_start, theta_start])
                
        for start in starts:
            if time.time() - start_time > time_limit * 0.8:
                break
            try:
                for beta in [50.0, 150.0, 400.0]:
                    if time.time() - start_time > time_limit * 0.8:
                        break
                    res_lbfgs = minimize(
                        lambda p, b=beta: -_soft_score_ring_params(n_out, n_in, p, b),
                        start,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
                    )
                    
                    exact_score = _score_ring_params(n_out, n_in, res_lbfgs.x)
                    if exact_score > best_score:
                        best_score = exact_score
                        best_params = res_lbfgs.x
                    start = res_lbfgs.x
            except Exception:
                pass
                
    # Phase 3: High-precision Nelder-Mead on exact objective
    if time.time() - start_time < time_limit * 0.98 and best_params is not None:
        try:
            res_nm = minimize(
                lambda p: -_score_ring_params(n_out, n_in, p),
                best_params,
                method='Nelder-Mead',
                options={'xatol': 1e-14, 'fatol': 1e-14, 'maxiter': 500, 'adaptive': True}
            )
            if -res_nm.fun > best_score:
                best_score = -res_nm.fun
                best_params = res_nm.x
        except Exception:
            pass

    if best_params is None:
        return 0.0, None
        
    return best_score, best_params

def _coordinate_descent_polish(points, deadline):
    """
    Coordinate descent: for each point, optimize its (x,y) position
    by moving it along a small grid or using 1D line search.
    """
    best_points = points.copy()
    best_score = _calculate_score(best_points)
    
    n = len(best_points)
    
    # Try moving each point in small steps
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    for step in step_sizes:
        if time.time() > deadline:
            break
        improved = True
        while improved:
            improved = False
            for idx in range(n):
                if time.time() > deadline:
                    break
                orig = best_points[idx].copy()
                best_local_score = best_score
                best_local_pt = orig.copy()
                
                # Try 8 directions + center
                for dx in [-step, 0.0, step]:
                    for dy in [-step, 0.0, step]:
                        if dx == 0.0 and dy == 0.0:
                            continue
                        candidate = best_points.copy()
                        candidate[idx] = orig + np.array([dx, dy])
                        s = _calculate_score(candidate)
                        if s > best_local_score + 1e-12:
                            best_local_score = s
                            best_local_pt = candidate[idx].copy()
                
                if best_local_score > best_score + 1e-12:
                    best_points[idx] = best_local_pt
                    best_score = best_local_score
                    improved = True
    
    return best_score, best_points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Candidate configurations (n_out, n_in)
    candidates = [
        (7, 7),
        (8, 6),
        (6, 8),
        (9, 5),
        (5, 9),
        (10, 4),
        (4, 10),
        (11, 3),
        (3, 11),
        (12, 2),
        (2, 12),
        (13, 1),
        (1, 13)
    ]
    
    best_overall_score = 0.0
    best_overall_points = None
    
    for n_out, n_in in candidates:
        if time.time() > deadline - 20:  # Reserve 20s for final polish
            break
            
        # Time budget for this candidate
        if (n_out, n_in) == (7, 7):
            t_limit = 35.0
        elif (n_out, n_in) in [(8, 6), (6, 8)]:
            t_limit = 15.0
        elif (n_out, n_in) in [(9, 5), (5, 9)]:
            t_limit = 10.0
        else:
            t_limit = 4.0
            
        score, params = _optimize_ring(n_out, n_in, t_limit)
        
        if score > best_overall_score and params is not None:
            best_overall_score = score
            r, theta = params
            angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
            x_out = np.cos(angles_out)
            y_out = np.sin(angles_out)
            
            angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + theta
            x_in = r * np.cos(angles_in)
            y_in = r * np.sin(angles_in)
            
            best_overall_points = np.vstack([
                np.column_stack((x_out, y_out)),
                np.column_stack((x_in, y_in))
            ])
    
    # Fallback if no valid points found
    if best_overall_points is None:
        n_out, n_in = 7, 7
        r, theta = 0.5, 0.0
        angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
        x_out = np.cos(angles_out)
        y_out = np.sin(angles_out)
        angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + theta
        x_in = r * np.cos(angles_in)
        y_in = r * np.sin(angles_in)
        best_overall_points = np.vstack([
            np.column_stack((x_out, y_out)),
            np.column_stack((x_in, y_in))
        ])
    
    # Final polish: coordinate descent for fine-tuning
    if time.time() < deadline - 2:
        cd_deadline = deadline - 2
        polished_score, polished_points = _coordinate_descent_polish(
            best_overall_points, cd_deadline
        )
        if polished_score > best_overall_score:
            best_overall_score = polished_score
            best_overall_points = polished_points
    
    return best_overall_points
