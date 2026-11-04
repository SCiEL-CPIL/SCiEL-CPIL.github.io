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

def _fourier_points(params):
    """
    Generate 14 points from Fourier series parameters.
    params: [r1c, r1s, r2c, r2s, a1c, a1s, a2c, a2s]
    - r1c, r1s: radial modulation at frequency 1 (breathing)
    - r2c, r2s: radial modulation at frequency 2
    - a1c, a1s: angular modulation at frequency 1
    - a2c, a2s: angular modulation at frequency 2
    """
    r1c, r1s, r2c, r2s, a1c, a1s, a2c, a2s = params
    base_angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    
    # Radial modulation
    radial = 1.0 + r1c * np.cos(base_angles) + r1s * np.sin(base_angles) \
                + r2c * np.cos(2*base_angles) + r2s * np.sin(2*base_angles)
    
    # Angular modulation
    angular = base_angles + a1c * np.cos(base_angles) + a1s * np.sin(base_angles) \
                + a2c * np.cos(2*base_angles) + a2s * np.sin(2*base_angles)
    
    # Clamp radial to maintain convexity
    radial = np.clip(radial, 0.5, 1.5)
    
    x = radial * np.cos(angular)
    y = radial * np.sin(angular)
    
    return np.column_stack((x, y))

def _fourier_score(params):
    """Score for Fourier parameterization."""
    points = _fourier_points(params)
    return _calculate_score(points)

def _soft_fourier_score(params, beta=100.0):
    """Soft score for Fourier parameterization."""
    points = _fourier_points(params)
    return -_soft_min_score(points, beta)

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

def _optimize_fourier(time_limit):
    """
    Optimize the Fourier parameterization.
    """
    start_time = time.time()
    
    # Bounds for Fourier coefficients
    # Radial: keep points in reasonable range
    # Angular: keep convexity (small perturbations)
    bounds = [
        (-0.15, 0.15),  # r1c
        (-0.15, 0.15),  # r1s
        (-0.10, 0.10),  # r2c
        (-0.10, 0.10),  # r2s
        (-0.05, 0.05),  # a1c
        (-0.05, 0.05),  # a1s
        (-0.03, 0.03),  # a2c
        (-0.03, 0.03),  # a2s
    ]
    
    best_score = 0.0
    best_params = np.zeros(8)
    
    # Phase 1: DE for global exploration
    if time.time() - start_time < time_limit * 0.6:
        try:
            res = differential_evolution(
                lambda p: -_fourier_score(p),
                bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-5,
                polish=False
            )
            
            if hasattr(res, 'x') and res.x is not None:
                s = _fourier_score(res.x)
                if s > best_score:
                    best_score = s
                    best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Local polish with soft surrogate
    if time.time() - start_time < time_limit * 0.9:
        # Multiple restarts around best point
        starts = [best_params.copy()]
        
        # Add some random perturbations
        rng = np.random.RandomState(42)
        for _ in range(10):
            perturbed = best_params + rng.uniform(-0.02, 0.02, size=8)
            starts.append(perturbed)
        
        for start in starts:
            if time.time() - start_time > time_limit * 0.9:
                break
            try:
                res_lbfgs = minimize(
                    _soft_fourier_score,
                    start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if res_lbfgs.x is not None:
                    exact_score = _fourier_score(res_lbfgs.x)
                    if exact_score > best_score:
                        best_score = exact_score
                        best_params = res_lbfgs.x.copy()
            except Exception:
                pass
        
        # Phase 3: Nelder-Mead polish
        if time.time() - start_time < time_limit:
            try:
                res_nm = minimize(
                    lambda p: -_fourier_score(p),
                    best_params,
                    method='Nelder-Mead',
                    options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 1000, 'adaptive': True}
                )
                if -res_nm.fun > best_score:
                    best_score = -res_nm.fun
                    best_params = res_nm.x.copy()
            except Exception:
                pass
    
    return best_score, best_params

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_overall_score = 0.0
    best_overall_points = None
    
    # Try ring configurations first (proven approach)
    ring_candidates = [
        (7, 7),
        (8, 6),
        (6, 8),
        (9, 5),
        (5, 9),
    ]
    
    for n_out, n_in in ring_candidates:
        if time.time() > deadline - 10:
            break
            
        if (n_out, n_in) == (7, 7):
            t_limit = 40.0
        elif (n_out, n_in) in [(8, 6), (6, 8)]:
            t_limit = 20.0
        else:
            t_limit = 10.0
            
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
    
    # Try Fourier parameterization with remaining time
    remaining_time = deadline - 5 - time.time()
    if remaining_time > 5:
        try:
            f_score, f_params = _optimize_fourier(remaining_time)
            if f_score > best_overall_score:
                best_overall_score = f_score
                best_overall_points = _fourier_points(f_params)
        except Exception:
            pass
    
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
