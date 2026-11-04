import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize, Bounds
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
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    
    # Calculate Minimum Triangle Area
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    
    min_area = 0.5 * np.min(np.abs(cross))
    
    return min_area / hull_area

def _soft_min_score(points, beta=50.0):
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
    # Stabilized: m - 1/beta * log(sum(exp(-beta * (areas - m))))
    max_area = np.max(areas)
    exp_terms = np.exp(-beta * (areas - max_area))
    sum_exp = np.sum(exp_terms)
    soft_min_area = max_area - np.log(sum_exp) / beta
    
    return soft_min_area / hull_area

def _generate_ring(n_out, n_in, r, theta):
    """
    Generate points for a two-ring configuration.
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

def _score_ring_params(n_out, n_in, params):
    """
    Objective function for differential evolution.
    params: [r, theta]
    """
    r, theta = params
    if r < 0 or r > 0.999:
        return 0.0
        
    points = _generate_ring(n_out, n_in, r, theta)
    return _calculate_score(points)

def _soft_score_ring_params(n_out, n_in, params, beta=50.0):
    """
    Soft objective function for gradient-based optimization.
    """
    r, theta = params
    if r < 0 or r > 0.999:
        return 0.0
        
    points = _generate_ring(n_out, n_in, r, theta)
    return _soft_min_score(points, beta)

def _optimize_ring(n_out, n_in, time_limit, deadline):
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
        if time_limit > 30:
            popsize = 15
            maxiter = 50
        elif time_limit > 15:
            popsize = 12
            maxiter = 40
        else:
            popsize = 10
            maxiter = 30
            
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
    
    # Phase 2: Local polish with Soft-Min L-BFGS-B
    if time.time() < deadline - 5.0:
        starts = []
        
        if best_params is not None:
            starts.append(best_params)
            
        for r_start in [0.3, 0.5, 0.7, 0.9]:
            for theta_start in [0.0, theta_max/2, theta_max/3, 2*theta_max/3]:
                starts.append([r_start, theta_start])
                
        for start in starts:
            if time.time() > deadline - 5.0:
                break
            try:
                beta = 50.0
                res_lbfgs = minimize(
                    lambda p: -_soft_score_ring_params(n_out, n_in, p, beta),
                    start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                exact_score = _score_ring_params(n_out, n_in, res_lbfgs.x)
                if exact_score > best_score:
                    best_score = exact_score
                    best_params = res_lbfgs.x
            except Exception:
                pass
                
    # Phase 3: High-precision polish with Nelder-Mead on exact objective
    if time.time() < deadline - 5.0 and best_params is not None:
        try:
            res_nm = minimize(
                lambda p: -_score_ring_params(n_out, n_in, p),
                best_params,
                method='Nelder-Mead',
                options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 500}
            )
            if -res_nm.fun > best_score:
                best_score = -res_nm.fun
                best_params = res_nm.x
        except Exception:
            pass

    if best_params is None:
        return 0.0, None
        
    return best_score, best_params

def _jittered_regular_polygon_score(params, seed=42):
    """
    Score for a jittered regular 14-gon.
    params: [scale, rotation, shear, noise_amplitude, noise_freq1, noise_freq2]
    """
    scale, rotation, shear, noise_amp, freq1, freq2 = params
    
    # Base regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    base_x = np.cos(angles)
    base_y = np.sin(angles)
    
    # Generate noise using a fixed low-dimensional Gaussian random field
    # We use a combination of sine waves with fixed random phases to create a smooth noise field
    rng = np.random.RandomState(seed)
    phases1 = rng.uniform(0, 2*np.pi, 3)
    phases2 = rng.uniform(0, 2*np.pi, 3)
    
    # Noise as a sum of harmonics
    noise = np.zeros(N)
    for k in range(3):
        noise += np.sin(freq1 * (k+1) * angles + phases1[k])
        noise += np.sin(freq2 * (k+1) * angles + phases2[k])
    
    # Radial displacement
    r = 1.0 + noise_amp * noise
    
    # Apply affine transformation: scale, rotation, shear
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    
    # Rotation
    cos_r = np.cos(rotation)
    sin_r = np.sin(rotation)
    x_rot = x * cos_r - y * sin_r
    y_rot = x * sin_r + y * cos_r
    
    # Shear (only in x direction for simplicity)
    x_shear = x_rot + shear * y_rot
    y_shear = y_rot
    
    # Scale
    x_final = scale * x_shear
    y_final = scale * y_shear
    
    points = np.column_stack((x_final, y_final))
    
    return _calculate_score(points)

def _soft_jittered_regular_polygon_score(params, beta=50.0, seed=42):
    """
    Soft score for jittered regular 14-gon.
    """
    scale, rotation, shear, noise_amp, freq1, freq2 = params
    
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    base_x = np.cos(angles)
    base_y = np.sin(angles)
    
    rng = np.random.RandomState(seed)
    phases1 = rng.uniform(0, 2*np.pi, 3)
    phases2 = rng.uniform(0, 2*np.pi, 3)
    
    noise = np.zeros(N)
    for k in range(3):
        noise += np.sin(freq1 * (k+1) * angles + phases1[k])
        noise += np.sin(freq2 * (k+1) * angles + phases2[k])
    
    r = 1.0 + noise_amp * noise
    
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    
    cos_r = np.cos(rotation)
    sin_r = np.sin(rotation)
    x_rot = x * cos_r - y * sin_r
    y_rot = x * sin_r + y * cos_r
    
    x_shear = x_rot + shear * y_rot
    y_shear = y_rot
    
    x_final = scale * x_shear
    y_final = scale * y_shear
    
    points = np.column_stack((x_final, y_final))
    
    return _soft_min_score(points, beta)

def _optimize_jittered_regular_polygon(time_limit, deadline):
    """
    Optimize the jittered regular 14-gon configuration.
    """
    # Parameters: [scale, rotation, shear, noise_amp, freq1, freq2]
    # Scale should be positive, rotation in [0, 2pi), shear small, noise_amp small
    bounds = [
        (0.5, 1.5),   # scale
        (0.0, 2*np.pi), # rotation
        (-0.5, 0.5),  # shear
        (0.0, 0.3),   # noise_amplitude
        (1.0, 5.0),   # freq1
        (1.0, 5.0)    # freq2
    ]
    
    best_score = 0.0
    best_params = None
    
    # Phase 1: Differential Evolution
    try:
        popsize = 20
        maxiter = 30
        
        res = differential_evolution(
            lambda p: -_jittered_regular_polygon_score(p),
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
    if time.time() < deadline - 5.0:
        starts = []
        
        if best_params is not None:
            starts.append(best_params)
            
        # Add heuristic starts
        starts.append([1.0, 0.0, 0.0, 0.0, 1.0, 1.0])  # Regular polygon
        starts.append([1.0, 0.0, 0.0, 0.1, 2.0, 3.0])  # Small noise
        starts.append([1.0, np.pi/14, 0.0, 0.05, 1.0, 2.0])  # Slight rotation and noise
        
        for start in starts:
            if time.time() > deadline - 5.0:
                break
            try:
                beta = 50.0
                res_lbfgs = minimize(
                    lambda p: -_soft_jittered_regular_polygon_score(p, beta),
                    start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                exact_score = _jittered_regular_polygon_score(res_lbfgs.x)
                if exact_score > best_score:
                    best_score = exact_score
                    best_params = res_lbfgs.x
            except Exception:
                pass
                
    # Phase 3: High-precision polish with Nelder-Mead
    if time.time() < deadline - 5.0 and best_params is not None:
        try:
            res_nm = minimize(
                lambda p: -_jittered_regular_polygon_score(p),
                best_params,
                method='Nelder-Mead',
                options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 500}
            )
            if -res_nm.fun > best_score:
                best_score = -res_nm.fun
                best_params = res_nm.x
        except Exception:
            pass

    if best_params is None:
        return 0.0, None
        
    return best_score, best_params

def _generate_jittered_points(params, seed=42):
    """
    Generate points from jittered regular polygon parameters.
    """
    scale, rotation, shear, noise_amp, freq1, freq2 = params
    
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    
    rng = np.random.RandomState(seed)
    phases1 = rng.uniform(0, 2*np.pi, 3)
    phases2 = rng.uniform(0, 2*np.pi, 3)
    
    noise = np.zeros(N)
    for k in range(3):
        noise += np.sin(freq1 * (k+1) * angles + phases1[k])
        noise += np.sin(freq2 * (k+1) * angles + phases2[k])
    
    r = 1.0 + noise_amp * noise
    
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    
    cos_r = np.cos(rotation)
    sin_r = np.sin(rotation)
    x_rot = x * cos_r - y * sin_r
    y_rot = x * sin_r + y * cos_r
    
    x_shear = x_rot + shear * y_rot
    y_shear = y_rot
    
    x_final = scale * x_shear
    y_final = scale * y_shear
    
    return np.column_stack((x_final, y_final))

def _coordinate_polish(points, deadline):
    """
    Polish all coordinates directly using L-BFGS-B with soft-min surrogate.
    """
    if time.time() > deadline - 5.0:
        return points, _calculate_score(points)
    
    best_points = points.copy()
    best_score = _calculate_score(points)
    
    def neg_soft_score(x):
        pts = x.reshape(-1, 2)
        return _soft_min_score(pts, beta=50.0)
    
    try:
        res = minimize(
            neg_soft_score,
            points.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        polished = res.x.reshape(-1, 2)
        polished_score = _calculate_score(polished)
        
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished
    except Exception:
        pass
    
    if time.time() < deadline - 5.0:
        try:
            res_nm = minimize(
                lambda x: -_calculate_score(x.reshape(-1, 2)),
                best_points.flatten(),
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 200}
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
    
    # Strategy 1: Ring-based configurations
    candidates = [
        (7, 7),
        (8, 6),
        (6, 8),
        (9, 5),
        (5, 9),
        (10, 4),
        (4, 10)
    ]
    
    for n_out, n_in in candidates:
        if time.time() > deadline - 20:
            break
            
        if (n_out, n_in) == (7, 7):
            t_limit = 50.0
        elif (n_out, n_in) in [(8, 6), (6, 8)]:
            t_limit = 20.0
        else:
            t_limit = 10.0
            
        score, params = _optimize_ring(n_out, n_in, t_limit, deadline)
        
        if score > best_overall_score:
            best_overall_score = score
            if params is not None:
                r, theta = params
                best_overall_points = _generate_ring(n_out, n_in, r, theta)
                
    # Strategy 2: Jittered regular polygon
    if time.time() < deadline - 15:
        t_limit = 25.0
        score, params = _optimize_jittered_regular_polygon(t_limit, deadline)
        
        if score > best_overall_score:
            best_overall_score = score
            if params is not None:
                best_overall_points = _generate_jittered_points(params)
                
    # Fallback
    if best_overall_points is None:
        n_out, n_in = 7, 7
        r, theta = 0.5, 0.0
        best_overall_points = _generate_ring(n_out, n_in, r, theta)
        
    # Final Coordinate Polish
    if time.time() < deadline - 5.0:
        best_overall_points, best_overall_score = _coordinate_polish(best_overall_points, deadline)
        
    return best_overall_points
