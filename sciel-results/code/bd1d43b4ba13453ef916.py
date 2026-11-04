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
    
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    
    min_area = 0.5 * np.min(np.abs(cross))
    
    return min_area / hull_area

def _soft_min_score(points, beta=100.0):
    """
    Smooth surrogate for min triangle area / hull area using log-sum-exp.
    Returns soft_min_area / hull_area.
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
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    max_area = np.max(areas)
    exp_terms = np.exp(-beta * (areas - max_area))
    sum_exp = np.sum(exp_terms)
    soft_min_area = max_area - np.log(sum_exp) / beta
    
    return soft_min_area / hull_area

def _generate_two_ring(n_out, n_in, r, theta):
    """
    Generate points for a two-ring configuration.
    Outer ring: n_out points on unit circle.
    Inner ring: n_in points on circle of radius r, rotated by theta.
    """
    angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
    x_out = np.cos(angles_out)
    y_out = np.sin(angles_out)
    
    angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + theta
    x_in = r * np.cos(angles_in)
    y_in = r * np.sin(angles_in)
    
    return np.vstack([
        np.column_stack((x_out, y_out)),
        np.column_stack((x_in, y_in))
    ])

def _generate_three_ring(r1, r2, theta1, theta2, n1=5, n2=5, n3=4):
    """
    Generate points for a three-ring configuration.
    Ring 1 (outer): n1 points on unit circle.
    Ring 2 (middle): n2 points on circle of radius r1, rotated by theta1.
    Ring 3 (inner): n3 points on circle of radius r2, rotated by theta2.
    """
    angles1 = np.linspace(0, 2*np.pi, n1, endpoint=False)
    x1 = np.cos(angles1)
    y1 = np.sin(angles1)
    
    angles2 = np.linspace(0, 2*np.pi, n2, endpoint=False) + theta1
    x2 = r1 * np.cos(angles2)
    y2 = r1 * np.sin(angles2)
    
    angles3 = np.linspace(0, 2*np.pi, n3, endpoint=False) + theta2
    x3 = r2 * np.cos(angles3)
    y3 = r2 * np.sin(angles3)
    
    return np.vstack([
        np.column_stack((x1, y1)),
        np.column_stack((x2, y2)),
        np.column_stack((x3, y3))
    ])

def _optimize_two_ring(n_out, n_in, time_budget, deadline):
    """
    Optimize a two-ring configuration.
    """
    start_time = time.time()
    
    g = np.gcd(n_out, n_in)
    theta_max = 2 * np.pi / g
    
    bounds = [(0.0, 0.999), (0.0, theta_max)]
    
    best_score = 0.0
    best_params = None
    
    # Phase 1: Differential Evolution
    try:
        popsize = 12
        maxiter = 30
        
        res = differential_evolution(
            lambda p: -_calculate_score(_generate_two_ring(n_out, n_in, p[0], p[1])),
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
    
    # Phase 2: Local polish with Soft-Min L-BFGS-B
    if time.time() < deadline - 10.0:
        starts = []
        
        if best_params is not None:
            starts.append(best_params)
            
        for r_start in [0.3, 0.5, 0.7]:
            for theta_start in [0.0, theta_max/2]:
                starts.append([r_start, theta_start])
                
        for start in starts:
            if time.time() > deadline - 10.0:
                break
            try:
                beta = 100.0
                res_lbfgs = minimize(
                    lambda p: -_soft_min_score(_generate_two_ring(n_out, n_in, p[0], p[1]), beta),
                    start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                exact_score = _calculate_score(_generate_two_ring(n_out, n_in, res_lbfgs.x[0], res_lbfgs.x[1]))
                if exact_score > best_score:
                    best_score = exact_score
                    best_params = res_lbfgs.x
            except Exception:
                pass
                
    # Phase 3: High-precision polish with Nelder-Mead on exact objective
    if time.time() < deadline - 10.0 and best_params is not None:
        try:
            res_nm = minimize(
                lambda p: -_calculate_score(_generate_two_ring(n_out, n_in, p[0], p[1])),
                best_params,
                method='Nelder-Mead',
                options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 200}
            )
            if -res_nm.fun > best_score:
                best_score = -res_nm.fun
                best_params = res_nm.x
        except Exception:
            pass

    if best_params is None:
        return 0.0, None
        
    return best_score, best_params

def _optimize_three_ring(time_budget, deadline):
    """
    Optimize a three-ring configuration.
    params: [r1, r2, theta1, theta2]
    """
    start_time = time.time()
    
    n1, n2, n3 = 5, 5, 4
    
    bounds = [
        (0.0, 0.999),  # r1 (middle ring radius)
        (0.0, 0.999),  # r2 (inner ring radius)
        (0.0, 2*np.pi/5),  # theta1
        (0.0, 2*np.pi/4)   # theta2
    ]
    
    best_score = 0.0
    best_params = None
    
    # Phase 1: Differential Evolution
    try:
        popsize = 10
        maxiter = 20
        
        res = differential_evolution(
            lambda p: -_calculate_score(_generate_three_ring(p[0], p[1], p[2], p[3])),
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
    
    # Phase 2: Local polish with Soft-Min L-BFGS-B
    if time.time() < deadline - 10.0:
        starts = []
        
        if best_params is not None:
            starts.append(best_params)
            
        # Heuristic starts
        starts.append([0.7, 0.4, 0.0, 0.0])
        starts.append([0.6, 0.3, np.pi/5, np.pi/4])
        starts.append([0.8, 0.5, np.pi/5, 0.0])
        
        for start in starts:
            if time.time() > deadline - 10.0:
                break
            try:
                beta = 100.0
                res_lbfgs = minimize(
                    lambda p: -_soft_min_score(_generate_three_ring(p[0], p[1], p[2], p[3]), beta),
                    start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                exact_score = _calculate_score(_generate_three_ring(res_lbfgs.x[0], res_lbfgs.x[1], res_lbfgs.x[2], res_lbfgs.x[3]))
                if exact_score > best_score:
                    best_score = exact_score
                    best_params = res_lbfgs.x
            except Exception:
                pass
                
    # Phase 3: High-precision polish with Nelder-Mead
    if time.time() < deadline - 10.0 and best_params is not None:
        try:
            res_nm = minimize(
                lambda p: -_calculate_score(_generate_three_ring(p[0], p[1], p[2], p[3])),
                best_params,
                method='Nelder-Mead',
                options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 200}
            )
            if -res_nm.fun > best_score:
                best_score = -res_nm.fun
                best_params = res_nm.x
        except Exception:
            pass

    if best_params is None:
        return 0.0, None
        
    return best_score, best_params

def _coordinate_polish(points, deadline):
    """
    Polish all coordinates directly using L-BFGS-B with soft-min surrogate.
    """
    if time.time() > deadline - 10.0:
        return points, _calculate_score(points)
    
    best_points = points.copy()
    best_score = _calculate_score(points)
    
    def neg_soft_score(x):
        pts = x.reshape(-1, 2)
        return -_soft_min_score(pts, beta=100.0)
    
    try:
        res = minimize(
            neg_soft_score,
            points.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        polished = res.x.reshape(-1, 2)
        polished_score = _calculate_score(polished)
        
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished
    except Exception:
        pass
    
    if time.time() < deadline - 10.0:
        try:
            res_nm = minimize(
                lambda x: -_calculate_score(x.reshape(-1, 2)),
                best_points.flatten(),
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 100}
            )
            polished = res_nm.x.reshape(-1, 2)
            polished_score = _calculate_score(polished)
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished
        except Exception:
            pass
            
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_overall_score = 0.0
    best_overall_points = None
    
    # Strategy 1: Two-ring configurations
    candidates = [
        (7, 7),
        (8, 6),
        (6, 8),
        (9, 5),
        (5, 9)
    ]
    
    for n_out, n_in in candidates:
        if time.time() > deadline - 30.0:
            break
            
        if (n_out, n_in) == (7, 7):
            t_limit = 40.0
        else:
            t_limit = 15.0
            
        score, params = _optimize_two_ring(n_out, n_in, t_limit, deadline)
        
        if score > best_overall_score:
            best_overall_score = score
            if params is not None:
                r, theta = params
                best_overall_points = _generate_two_ring(n_out, n_in, r, theta)
                
    # Strategy 2: Three-ring configuration
    if time.time() < deadline - 20.0:
        score, params = _optimize_three_ring(20.0, deadline)
        
        if score > best_overall_score:
            best_overall_score = score
            if params is not None:
                r1, r2, theta1, theta2 = params
                best_overall_points = _generate_three_ring(r1, r2, theta1, theta2)
                
    # Fallback
    if best_overall_points is None:
        best_overall_points = _generate_two_ring(7, 7, 0.5, 0.0)
        
    # Final Coordinate Polish
    if time.time() < deadline - 5.0:
        best_overall_points, best_overall_score = _coordinate_polish(best_overall_points, deadline)
        
    return best_overall_points
