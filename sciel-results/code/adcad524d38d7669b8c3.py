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
    Smooth surrogate for min triangle area using log-sum-exp.
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
    
    abs_cross = np.abs(cross)
    
    # Soft min via log-sum-exp for numerical stability
    max_val = np.max(abs_cross)
    exp_vals = np.exp(-beta * (abs_cross - max_val))
    soft_min_val = max_val + np.log(np.sum(exp_vals)) / beta
    
    return 0.5 * soft_min_val / hull_area

def _score_ring_params(n_out, n_in, params):
    """
    Objective function for differential evolution.
    params: [r, theta]
    """
    r, theta = params
    if r < 0 or r > 0.999:
        return 0.0
        
    angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
    x_out = np.cos(angles_out)
    y_out = np.sin(angles_out)
    
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
    Smooth objective for gradient-based optimization (minimize negative).
    """
    r, theta = params
    if r < 0 or r > 0.999:
        return 1.0
        
    angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
    x_out = np.cos(angles_out)
    y_out = np.sin(angles_out)
    
    angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + theta
    x_in = r * np.cos(angles_in)
    y_in = r * np.sin(angles_in)
    
    points = np.vstack([
        np.column_stack((x_out, y_out)),
        np.column_stack((x_in, y_in))
    ])
    
    return -_soft_min_score(points, beta)

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
    
    # Phase 1: DE for global exploration
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
            best_params = res.x.copy()
        elif hasattr(res, 'x') and res.x is not None:
            s = _score_ring_params(n_out, n_in, res.x)
            if s > best_score:
                best_score = s
                best_params = res.x.copy()
    except Exception:
        pass
    
    # Phase 2: Local polish with smooth surrogate + L-BFGS-B
    if time.time() - start_time < time_limit * 0.7:
        starts = []
        
        if best_params is not None:
            starts.append(best_params.copy())
            
        r_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        theta_values = [0.0, theta_max/4, theta_max/2, 3*theta_max/4]
        
        for r_start in r_values:
            for theta_start in theta_values:
                starts.append([r_start, theta_start])
        
        if best_params is not None:
            for dr in [-0.05, 0.05]:
                for dtheta in [-0.1, 0.1]:
                    r_p = np.clip(best_params[0] + dr, 0.01, 0.998)
                    th_p = (best_params[1] + dtheta) % theta_max
                    starts.append([r_p, th_p])
        
        for start in starts:
            if time.time() - start_time > time_limit * 0.7:
                break
            try:
                res_lbfgs = minimize(
                    lambda p: _soft_score_ring_params(n_out, n_in, p, beta=100.0),
                    start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if res_lbfgs.x is not None:
                    exact_score = _score_ring_params(n_out, n_in, res_lbfgs.x)
                    if exact_score > best_score:
                        best_score = exact_score
                        best_params = res_lbfgs.x.copy()
            except Exception:
                pass
        
        # Phase 3: Final Nelder-Mead polish on best point
        if best_params is not None and time.time() - start_time < time_limit:
            try:
                res_nm = minimize(
                    lambda p: -_score_ring_params(n_out, n_in, p),
                    best_params,
                    method='Nelder-Mead',
                    options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 500, 'adaptive': True}
                )
                if -res_nm.fun > best_score:
                    best_score = -res_nm.fun
                    best_params = res_nm.x.copy()
            except Exception:
                pass
    
    if best_params is None:
        return 0.0, None
        
    return best_score, best_params

def _double_ring_points(params):
    """
    Generate 14 points as a double-ring with 7-fold symmetry.
    Each vertex of a regular heptagon is replaced by a pair of points
    separated radially by an optimized offset.
    
    params: [r_inner, r_outer, theta_offset]
    - r_inner: radius of inner ring
    - r_outer: radius of outer ring  
    - theta_offset: angular offset between the two rings
    """
    r_inner, r_outer, theta_offset = params
    
    angles_base = np.linspace(0, 2*np.pi, 7, endpoint=False)
    
    # Outer ring at base angles
    x_out = r_outer * np.cos(angles_base)
    y_out = r_outer * np.sin(angles_base)
    
    # Inner ring at base angles + theta_offset
    angles_in = angles_base + theta_offset
    x_in = r_inner * np.cos(angles_in)
    y_in = r_inner * np.sin(angles_in)
    
    # Interleave: for each vertex i, place outer[i], inner[i]
    points = np.zeros((14, 2))
    for i in range(7):
        points[2*i] = [x_out[i], y_out[i]]
        points[2*i+1] = [x_in[i], y_in[i]]
    
    return points

def _score_double_ring(params):
    """Score for double-ring configuration."""
    points = _double_ring_points(params)
    return _calculate_score(points)

def _soft_score_double_ring(params, beta=100.0):
    """Smooth score for double-ring configuration."""
    points = _double_ring_points(params)
    return -_soft_min_score(points, beta)

def _optimize_double_ring(time_limit):
    """
    Optimize the double-ring configuration with 7-fold symmetry.
    """
    start_time = time.time()
    
    # Bounds: r_inner in [0.1, 0.95], r_outer in [0.5, 1.5], theta in [0, 2*pi/7]
    theta_max = 2 * np.pi / 7
    bounds = [(0.1, 0.95), (0.5, 1.5), (0.0, theta_max)]
    
    best_score = 0.0
    best_params = None
    
    # Phase 1: DE for global exploration
    try:
        res = differential_evolution(
            lambda p: -_score_double_ring(p),
            bounds,
            seed=42,
            maxiter=30,
            popsize=15,
            tol=1e-5,
            polish=False
        )
        
        if res.success:
            best_score = -res.fun
            best_params = res.x.copy()
        elif hasattr(res, 'x') and res.x is not None:
            s = _score_double_ring(res.x)
            if s > best_score:
                best_score = s
                best_params = res.x.copy()
    except Exception:
        pass
    
    # Phase 2: Local polish with multiple starts
    if time.time() - start_time < time_limit * 0.6:
        starts = []
        
        if best_params is not None:
            starts.append(best_params.copy())
        
        # Grid of starting points
        for r_in in [0.3, 0.5, 0.7, 0.85]:
            for r_out in [0.9, 1.1, 1.3]:
                for th in [0.0, theta_max/4, theta_max/2, 3*theta_max/4]:
                    starts.append([r_in, r_out, th])
        
        for start in starts:
            if time.time() - start_time > time_limit * 0.6:
                break
            try:
                res_lbfgs = minimize(
                    lambda p: _soft_score_double_ring(p, beta=100.0),
                    start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if res_lbfgs.x is not None:
                    exact_score = _score_double_ring(res_lbfgs.x)
                    if exact_score > best_score:
                        best_score = exact_score
                        best_params = res_lbfgs.x.copy()
            except Exception:
                pass
        
        # Phase 3: Nelder-Mead polish
        if best_params is not None and time.time() - start_time < time_limit:
            try:
                res_nm = minimize(
                    lambda p: -_score_double_ring(p),
                    best_params,
                    method='Nelder-Mead',
                    options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 500, 'adaptive': True}
                )
                if -res_nm.fun > best_score:
                    best_score = -res_nm.fun
                    best_params = res_nm.x.copy()
            except Exception:
                pass
    
    if best_params is None:
        return 0.0, None
        
    return best_score, best_params

def _polish_points(points, time_limit):
    """
    Final coordinate-level polish on all 28 coordinates.
    This can escape local optima of the symmetric parameterization.
    """
    start_time = time.time()
    
    best_score = _calculate_score(points)
    best_points = points.copy()
    
    # Use L-BFGS-B with soft-min surrogate
    try:
        def objective(x):
            p = x.reshape(N, 2)
            return -_soft_min_score(p, beta=200.0)
        
        res = minimize(
            objective,
            points.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        
        if res.x is not None:
            polished = res.x.reshape(N, 2)
            s = _calculate_score(polished)
            if s > best_score:
                best_score = s
                best_points = polished
    except Exception:
        pass
    
    # Nelder-Mead polish with tighter tolerances
    if time.time() - start_time < time_limit:
        try:
            def objective_nm(x):
                p = x.reshape(N, 2)
                return -_calculate_score(p)
            
            res_nm = minimize(
                objective_nm,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 2000, 'adaptive': True}
            )
            
            if res_nm.x is not None:
                polished = res_nm.x.reshape(N, 2)
                s = _calculate_score(polished)
                if s > best_score:
                    best_score = s
                    best_points = polished
        except Exception:
            pass
    
    return best_score, best_points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
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
        if time.time() > deadline - 10:
            break
            
        if (n_out, n_in) == (7, 7):
            t_limit = 60.0
        elif (n_out, n_in) in [(8, 6), (6, 8)]:
            t_limit = 25.0
        elif (n_out, n_in) in [(9, 5), (5, 9)]:
            t_limit = 15.0
        elif (n_out, n_in) in [(10, 4), (4, 10)]:
            t_limit = 8.0
        else:
            t_limit = 4.0
            
        score, params = _optimize_ring(n_out, n_in, t_limit)
        
        if score > best_overall_score:
            best_overall_score = score
            if params is not None:
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
                
    # Try double-ring configuration
    if time.time() < deadline - 15:
        dr_limit = min(30.0, deadline - time.time() - 10)
        dr_score, dr_params = _optimize_double_ring(dr_limit)
        
        if dr_score > best_overall_score:
            best_overall_score = dr_score
            best_overall_points = _double_ring_points(dr_params)
    
    # Final coordinate-level polish
    if best_overall_points is not None and time.time() < deadline - 5:
        polish_limit = min(20.0, deadline - time.time() - 3)
        pol_score, pol_points = _polish_points(best_overall_points, polish_limit)
        
        if pol_score > best_overall_score:
            best_overall_score = pol_score
            best_overall_points = pol_points
    
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
        
    return best_overall_points
