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

def _compute_ring_areas(n_out, n_in, r, theta):
    """
    Compute all triangle areas for a two-ring configuration.
    Returns array of areas for all C(14,3) triples.
    """
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
    
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    
    return 0.5 * np.abs(cross)

def _optimize_77_analytic(theta):
    """
    For fixed theta, find optimal r that maximizes min triangle area.
    Uses the fact that we can compute all areas as functions of r.
    """
    n_out, n_in = 7, 7
    
    # Compute areas for a grid of r values to find approximate optimum
    r_grid = np.linspace(0.1, 0.95, 40)
    best_r = 0.5
    best_min_area = 0.0
    
    for r in r_grid:
        areas = _compute_ring_areas(n_out, n_in, r, theta)
        min_area = np.min(areas)
        if min_area > best_min_area:
            best_min_area = min_area
            best_r = r
    
    # Refine with golden section search
    a, b = max(0.05, best_r - 0.08), min(0.98, best_r + 0.08)
    gr = (np.sqrt(5) - 1) / 2
    
    for _ in range(20):
        m1 = b - gr * (b - a)
        m2 = a + gr * (b - a)
        
        areas1 = _compute_ring_areas(n_out, n_in, m1, theta)
        areas2 = _compute_ring_areas(n_out, n_in, m2, theta)
        
        if np.min(areas1) < np.min(areas2):
            a = m1
        else:
            b = m2
    
    r_opt = (a + b) / 2
    areas = _compute_ring_areas(n_out, n_in, r_opt, theta)
    min_area = np.min(areas)
    
    # Compute hull area (for 7-7, hull is the outer heptagon if r < 1)
    if r_opt < 0.999:
        # Hull is the outer regular heptagon
        hull_area = 7 * 0.5 * np.sin(2 * np.pi / 7)
    else:
        # Need to compute actual hull
        angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
        x_out = np.cos(angles_out)
        y_out = np.sin(angles_out)
        angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + theta
        x_in = r_opt * np.cos(angles_in)
        y_in = r_opt * np.sin(angles_in)
        points = np.vstack([
            np.column_stack((x_out, y_out)),
            np.column_stack((x_in, y_in))
        ])
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
        except:
            hull_area = 7 * 0.5 * np.sin(2 * np.pi / 7)
    
    return min_area / hull_area, r_opt

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

def _score_77_theta(theta):
    """
    For 7-7 ring configuration, score as function of theta only.
    Uses analytic r optimization.
    """
    score, r_opt = _optimize_77_analytic(theta)
    return score

def _optimize_77_1d(time_limit):
    """
    1D optimization over theta for 7-7 configuration.
    Uses golden section search with analytic r optimization.
    """
    start_time = time.time()
    
    # The symmetry group has order 7, so theta in [0, 2*pi/7)
    theta_max = 2 * np.pi / 7
    
    # Coarse grid search first
    n_grid = 25
    theta_grid = np.linspace(0, theta_max, n_grid, endpoint=False)
    
    best_score = 0.0
    best_theta = 0.0
    
    for theta in theta_grid:
        if time.time() - start_time > time_limit * 0.35:
            break
        score = _score_77_theta(theta)
        if score > best_score:
            best_score = score
            best_theta = theta
    
    # Golden section search to refine
    # Find bracket around best_theta
    step = theta_max / n_grid
    a = max(0, best_theta - step)
    b = min(theta_max, best_theta + step)
    
    gr = (np.sqrt(5) - 1) / 2
    max_iter = 50
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit * 0.8:
            break
        
        m1 = b - gr * (b - a)
        m2 = a + gr * (b - a)
        
        s1 = _score_77_theta(m1)
        s2 = _score_77_theta(m2)
        
        if s1 < s2:
            a = m1
        else:
            b = m2
    
    theta_opt = (a + b) / 2
    
    # Get final score and r
    final_score, r_opt = _optimize_77_analytic(theta_opt)
    
    # Build points
    n_out, n_in = 7, 7
    angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
    x_out = np.cos(angles_out)
    y_out = np.sin(angles_out)
    
    angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + theta_opt
    x_in = r_opt * np.cos(angles_in)
    y_in = r_opt * np.sin(angles_in)
    
    points = np.vstack([
        np.column_stack((x_out, y_out)),
        np.column_stack((x_in, y_in))
    ])
    
    return final_score, points

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

def _score_three_ring(params):
    """
    Score for a three-ring configuration: [r1, r2, theta1, theta2]
    Ring 0: 5 points on unit circle (fixed)
    Ring 1: 4 points on circle of radius r1, rotated by theta1
    Ring 2: 5 points on circle of radius r2, rotated by theta2
    """
    r1, r2, theta1, theta2 = params
    if r1 < 0 or r1 > 0.999 or r2 < 0 or r2 > 0.999:
        return 0.0
    
    n0, n1, n2 = 5, 4, 5
    
    angles0 = np.linspace(0, 2*np.pi, n0, endpoint=False)
    angles1 = np.linspace(0, 2*np.pi, n1, endpoint=False) + theta1
    angles2 = np.linspace(0, 2*np.pi, n2, endpoint=False) + theta2
    
    points = np.vstack([
        np.column_stack((np.cos(angles0), np.sin(angles0))),
        np.column_stack((r1 * np.cos(angles1), r1 * np.sin(angles1))),
        np.column_stack((r2 * np.cos(angles2), r2 * np.sin(angles2)))
    ])
    
    return _calculate_score(points)

def _soft_score_three_ring(params, beta=100.0):
    """Soft score for three-ring configuration (returns negative for minimization)."""
    r1, r2, theta1, theta2 = params
    if r1 < 0 or r1 > 0.999 or r2 < 0 or r2 > 0.999:
        return 1.0
    
    n0, n1, n2 = 5, 4, 5
    
    angles0 = np.linspace(0, 2*np.pi, n0, endpoint=False)
    angles1 = np.linspace(0, 2*np.pi, n1, endpoint=False) + theta1
    angles2 = np.linspace(0, 2*np.pi, n2, endpoint=False) + theta2
    
    points = np.vstack([
        np.column_stack((np.cos(angles0), np.sin(angles0))),
        np.column_stack((r1 * np.cos(angles1), r1 * np.sin(angles1))),
        np.column_stack((r2 * np.cos(angles2), r2 * np.sin(angles2)))
    ])
    
    return -_soft_min_score(points, beta)

def _optimize_three_ring(time_limit):
    """Optimize the 5-4-5 three-ring configuration."""
    start_time = time.time()
    
    bounds = [(0.1, 0.95), (0.1, 0.95), (0.0, 2*np.pi/5), (0.0, 2*np.pi/5)]
    
    best_score = 0.0
    best_params = None
    
    # Phase 1: DE for global exploration
    try:
        res = differential_evolution(
            lambda p: -_score_three_ring(p),
            bounds,
            seed=42,
            maxiter=60,
            popsize=15,
            tol=1e-5,
            polish=False
        )
        
        if res.success:
            best_score = -res.fun
            best_params = res.x.copy()
        elif hasattr(res, 'x') and res.x is not None:
            s = _score_three_ring(res.x)
            if s > best_score:
                best_score = s
                best_params = res.x.copy()
    except Exception:
        pass
    
    # Phase 2: Local polish
    if best_params is not None and time.time() - start_time < time_limit * 0.7:
        try:
            res_lbfgs = minimize(
                lambda p: _soft_score_three_ring(p, beta=100.0),
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if res_lbfgs.x is not None:
                exact_score = _score_three_ring(res_lbfgs.x)
                if exact_score > best_score:
                    best_score = exact_score
                    best_params = res_lbfgs.x.copy()
        except Exception:
            pass
        
        # Phase 3: Nelder-Mead polish
        if best_params is not None and time.time() - start_time < time_limit:
            try:
                res_nm = minimize(
                    lambda p: -_score_three_ring(p),
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
    
    # Build points
    r1, r2, theta1, theta2 = best_params
    n0, n1, n2 = 5, 4, 5
    
    angles0 = np.linspace(0, 2*np.pi, n0, endpoint=False)
    angles1 = np.linspace(0, 2*np.pi, n1, endpoint=False) + theta1
    angles2 = np.linspace(0, 2*np.pi, n2, endpoint=False) + theta2
    
    points = np.vstack([
        np.column_stack((np.cos(angles0), np.sin(angles0))),
        np.column_stack((r1 * np.cos(angles1), r1 * np.sin(angles1))),
        np.column_stack((r2 * np.cos(angles2), r2 * np.sin(angles2)))
    ])
    
    return best_score, points

def construct():
    start_time = time.time()
    deadline = start_time + 147.0
    
    best_overall_score = 0.0
    best_overall_points = None
    
    # Phase 1: 1D optimization for 7-7 (fastest, most promising)
    t_limit_77 = min(50.0, deadline - start_time - 5.0)
    if t_limit_77 > 5.0:
        score_77, points_77 = _optimize_77_1d(t_limit_77)
        if score_77 > best_overall_score:
            best_overall_score = score_77
            best_overall_points = points_77
    
    # Phase 2: Try three-ring 5-4-5 configuration
    remaining = deadline - time.time() - 3.0
    if remaining > 15.0:
        t_limit_3ring = min(30.0, remaining - 5.0)
        if t_limit_3ring > 5.0:
            score_3, points_3 = _optimize_three_ring(t_limit_3ring)
            if score_3 > best_overall_score:
                best_overall_score = score_3
                best_overall_points = points_3
    
    # Phase 3: Try other two-ring configurations if time remains
    remaining = deadline - time.time() - 3.0
    if remaining > 10.0:
        candidates = [
            (8, 6),
            (6, 8),
            (9, 5),
            (5, 9),
        ]
        
        for n_out, n_in in candidates:
            if time.time() > deadline - 3:
                break
            
            if remaining > 15.0:
                t_limit = min(15.0, remaining - 5.0)
            elif remaining > 8.0:
                t_limit = min(8.0, remaining - 3.0)
            else:
                break
            
            if t_limit < 4.0:
                break
            
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
            
            remaining = deadline - time.time() - 3.0
    
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
