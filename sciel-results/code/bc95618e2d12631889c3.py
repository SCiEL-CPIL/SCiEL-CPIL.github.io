import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
import time

N = 14
SOFT_BETA = 50.0  # Temperature for soft-min; higher = sharper min approximation

# Precompute triple indices for N=14
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
])

# Global deadline for time-aware DE callback
_DE_DEADLINE = [0.0]

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
    # points shape (N, 2)
    # TRIPLE_INDICES shape (M, 3)
    
    # Extract points for all triples
    # This creates a (M, 3, 2) array
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]
    
    # Compute cross product (pj - pi) x (pk - pi)
    # (xj - xi)(yk - yi) - (yj - yi)(xk - xi)
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    
    min_area = 0.5 * np.min(np.abs(cross))
    
    return min_area / hull_area

def _soft_min_score(points, beta=SOFT_BETA):
    """
    Smooth surrogate: -soft_min of triangle areas / hull area.
    Returns a smooth value that approximates -min_area/hull_area.
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
    
    # Soft-min via log-sum-exp: soft_min(a) ≈ (1/beta) * log(sum(exp(-beta * a)))
    # This is smooth and approximates min(a) as beta -> inf
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    soft_min_val = (1.0 / beta) * (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled))))
    
    return soft_min_val / hull_area

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

def _time_aware_de_callback(xk, convergence):
    """
    Callback for DE to check time. Returns True to stop if time is running out.
    """
    if time.time() > _DE_DEADLINE[0] - 3.0:
        return True
    return False

def _optimize_ring(n_out, n_in, time_limit, start_time=0, deadline=0):
    """
    Optimize a specific ring configuration.
    """
    global _DE_DEADLINE
    if start_time == 0:
        start_time = time.time()
    if deadline == 0:
        deadline = start_time + time_limit

    g = np.gcd(n_out, n_in)
    theta_max = 2 * np.pi / g
    
    bounds = [(0.0, 0.999), (0.0, theta_max)]
    
    best_score = 0.0
    best_params = None
    
    popsize = 10
    maxiter = 30
    
    if time_limit > 30:
        popsize = 15
        maxiter = 50
    elif time_limit > 15:
        popsize = 12
        maxiter = 40

    _DE_DEADLINE = [deadline]
    try:
        res = differential_evolution(
            lambda p: -_score_ring_params(n_out, n_in, p),
            bounds,
            seed=42,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-5,
            polish=True,
            callback=_time_aware_de_callback
        )
        
        if res.success:
            best_score = -res.fun
            best_params = res.x
    except Exception:
        pass
    
    if time.time() < deadline - 5.0:
        starts = []
        for r_start in [0.3, 0.5, 0.7, 0.9, 0.95]:
            for theta_start in [0.0, theta_max/2, theta_max/3, 2*theta_max/3, theta_max/4, 3*theta_max/4]:
                starts.append([r_start, theta_start])
                
        for start in starts:
            if time.time() > deadline - 5.0:
                break
            try:
                res_nm = minimize(
                    lambda p: -_score_ring_params(n_out, n_in, p),
                    start,
                    method='Nelder-Mead',
                    options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 200}
                )
                if -res_nm.fun > best_score:
                    best_score = -res_nm.fun
                    best_params = res_nm.x
            except Exception:
                pass
                
    if best_params is None:
        return 0.0, None
        
    return best_score, best_params

def _score_2p_params(params, n_out, n_in):
    """
    Objective for 2-parameter optimization: r and theta.
    This is the same as _score_ring_params but kept for clarity or potential future use.
    """
    return _score_ring_params(n_out, n_in, params)

def _soft_min_ring_score(n_out, n_in, params):
    """
    Smooth surrogate objective for ring parameters using soft-min.
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
    
    return _soft_min_score(points)

def _coordinate_polish(points, n_out, n_in, time_budget, deadline):
    """
    Polish all 28 coordinates directly using L-BFGS-B with soft-min surrogate.
    This can escape local optima of the ring parameterization.
    """
    if time.time() > deadline - 3.0:
        return points, _calculate_score(points)
    
    best_points = points.copy()
    best_score = _calculate_score(points)
    
    # Use soft-min for smooth gradient, then verify with exact min
    def neg_soft_score(x):
        pts = x.reshape(-1, 2)
        return _soft_min_score(pts, beta=SOFT_BETA)
    
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
            best_points = polished
            best_score = polished_score
            
    except Exception:
        pass
    
    # Second polish with higher beta for sharper min
    if time.time() < deadline - 3.0:
        def neg_soft_score_high(x):
            pts = x.reshape(-1, 2)
            return _soft_min_score(pts, beta=SOFT_BETA * 5)
        
        try:
            res2 = minimize(
                neg_soft_score_high,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            polished2 = res2.x.reshape(-1, 2)
            polished2_score = _calculate_score(polished2)
            
            if polished2_score > best_score:
                best_points = polished2
                best_score = polished2_score
                
        except Exception:
            pass
    
    return best_points, best_score

def _optimize_2p(n_out, n_in, time_limit, start_time, deadline):
    """
    Optimize the 2-parameter ring configuration (r, theta).
    """
    global _DE_DEADLINE
    g = np.gcd(n_out, n_in)
    theta_max = 2 * np.pi / g
    
    bounds = [(0.0, 0.999), (0.0, theta_max)]
    
    best_score = 0.0
    best_params = None
    best_points = None
    
    # Phase 1: Differential Evolution with soft-min surrogate for smoothness
    popsize = 10
    maxiter = 25
    
    if time_limit > 30:
        popsize = 15
        maxiter = 40
    elif time_limit > 15:
        popsize = 12
        maxiter = 35

    _DE_DEADLINE = [deadline]
    try:
        res = differential_evolution(
            lambda p: _soft_min_ring_score(n_out, n_in, p),
            bounds,
            seed=42,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-5,
            polish=True,
            callback=_time_aware_de_callback
        )
        
        if res.success:
            best_params = res.x
            r, theta = best_params
            best_points = _get_points_2p(n_out, n_in, r, theta)
            best_score = _calculate_score(best_points)
    except Exception:
        pass
    
    # Phase 2: Local polish with multiple starts using exact score
    if time.time() < deadline - 8.0:
        starts = []
        for r_start in [0.3, 0.5, 0.7, 0.9, 0.95]:
            for theta_start in [0.0, theta_max/2, theta_max/3, 2*theta_max/3]:
                starts.append([r_start, theta_start])
                
        for start in starts:
            if time.time() > deadline - 8.0:
                break
            try:
                res_nm = minimize(
                    lambda p: -_score_2p_params(p, n_out, n_in),
                    start,
                    method='Nelder-Mead',
                    options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 150}
                )
                if -res_nm.fun > best_score:
                    best_score = -res_nm.fun
                    best_params = res_nm.x
                    r, theta = best_params
                    best_points = _get_points_2p(n_out, n_in, r, theta)
            except Exception:
                pass
    
    # Phase 3: Coordinate-level polish to escape ring parameterization
    if best_points is not None and time.time() < deadline - 5.0:
        coord_time = min(10.0, deadline - 5.0 - time.time())
        best_points, best_score = _coordinate_polish(best_points, n_out, n_in, coord_time, deadline)
        
    if best_points is None:
        return 0.0, None
        
    return best_score, best_points.flatten()

def _get_points_2p(n_out, n_in, r, theta):
    """
    Generate points for the 2-parameter ring configuration.
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

def _generate_jittered_points(params):
    """
    Generate points from jittered regular polygon model.
    params: [M(6), delta(14)]
    M is flattened 2x3 affine matrix (row-major: a,b,c, d,e,f)
    delta is radial offsets for 14 vertices.
    """
    M = params[:6].reshape(2, 3)
    delta = params[6:20]
    
    # Base regular 14-gon vertices
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    x_base = np.cos(angles)
    y_base = np.sin(angles)
    base_points = np.column_stack((x_base, y_base))
    
    # Apply radial jitter
    # P_i = (1 + delta_i) * base_i
    radial_scale = 1.0 + delta
    jittered_points = base_points * radial_scale[:, np.newaxis]
    
    # Apply affine transform
    # P_final = P_jittered @ M[:2].T + M[:, 2]
    # M is 2x3. 
    # x' = a*x + b*y + c
    # y' = d*x + e*y + f
    points = jittered_points @ M[:2].T + M[:, 2]
    
    return points

def _soft_min_jittered_score(params, beta=SOFT_BETA):
    """
    Soft-min score for jittered model.
    """
    points = _generate_jittered_points(params)
    return _soft_min_score(points, beta=beta)

def _optimize_jittered(time_limit, deadline):
    """
    Optimize the jittered regular polygon model.
    """
    start_time = time.time()
    
    # Initial parameters
    # M = Identity + small shear
    # delta = 0
    best_params = np.zeros(20)
    best_params[0] = 1.0
    best_params[4] = 1.0
    
    best_points = _generate_jittered_points(best_params)
    best_score = _calculate_score(best_points)
    
    # Nelder-Mead on 20 parameters
    # We use soft-min for the objective to guide the search
    def neg_soft_obj(p):
        return _soft_min_jittered_score(p, beta=SOFT_BETA)
    
    try:
        # Run multiple restarts with different initial shears/jitters
        restarts = 5
        for i in range(restarts):
            if time.time() > deadline - 5.0:
                break
            
            # Perturb initial M and delta
            p_init = np.zeros(20)
            p_init[0] = 1.0
            p_init[4] = 1.0
            
            # Random small shear
            if i > 0:
                p_init[1] = np.random.uniform(-0.2, 0.2) # b
                p_init[3] = np.random.uniform(-0.2, 0.2) # d
                
                # Random small jitter
                p_init[6:] = np.random.uniform(-0.1, 0.1, 14)
            
            res = minimize(
                neg_soft_obj,
                p_init,
                method='Nelder-Mead',
                options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 300}
            )
            
            if -res.fun > best_score:
                best_score = -res.fun
                best_params = res.x
                best_points = _generate_jittered_points(best_params)
                
    except Exception:
        pass
    
    # Polish the best result with exact score and higher beta soft-min
    if time.time() < deadline - 3.0:
        best_points = _generate_jittered_points(best_params)
        best_score = _calculate_score(best_points)
        
        # L-BFGS-B on coordinates? 
        # The parameterization is 20D, coordinates are 28D.
        # Let's stick to parameter polish with higher beta.
        def neg_soft_obj_high(p):
            return _soft_min_jittered_score(p, beta=SOFT_BETA * 10)
            
        try:
            res_polish = minimize(
                neg_soft_obj_high,
                best_params,
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 100}
            )
            if -res_polish.fun > best_score:
                best_score = -res_polish.fun
                best_params = res_polish.x
                best_points = _generate_jittered_points(best_params)
                best_score = _calculate_score(best_points)
        except Exception:
            pass
            
    return best_score, best_points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s kill
    
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
    
    # Run ring strategies first (they tend to give better scores)
    # Give more time to likely candidates
    for n_out, n_in in candidates:
        if time.time() > deadline - 25:  # Reserve 25s for jittered + final
            break
            
        if (n_out, n_in) == (7, 7):
            t_limit = 40.0
        elif (n_out, n_in) in [(8, 6), (6, 8)]:
            t_limit = 20.0
        elif (n_out, n_in) in [(9, 5), (5, 9)]:
            t_limit = 12.0
        else:
            t_limit = 6.0
            
        score, params = _optimize_2p(n_out, n_in, t_limit, start_time, deadline)
        
        if score > best_overall_score:
            best_overall_score = score
            if params is not None:
                best_overall_points = params.reshape(-1, 2)
                actual_score = _calculate_score(best_overall_points)
                best_overall_score = actual_score
    
    # Run jittered strategy with remaining time
    if time.time() < deadline - 5.0:
        jitter_deadline = deadline - 5.0
        jitter_time = jitter_deadline - time.time()
        if jitter_time > 5.0:
            try:
                j_score, j_points = _optimize_jittered(jitter_time, jitter_deadline)
                if j_score > best_overall_score:
                    best_overall_score = j_score
                    best_overall_points = j_points
            except Exception:
                pass
    
    # Fallback if no valid points found
    if best_overall_points is None:
        n_out, n_in = 7, 7
        r, theta = 0.5, 0.0
        best_overall_points = _get_points_2p(n_out, n_in, r, theta)
        
    return best_overall_points
