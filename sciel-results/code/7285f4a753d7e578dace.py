import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize, minimize_scalar
import time

N = 14
SOFT_BETA = 50.0  # Temperature for soft-min; higher = sharper min approximation

# Precompute triple indices for N=14
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
])

# Precompute for each point index, which triples it participates in
_TRIPLE_POINT_TO_IDX = [np.where((TRIPLE_INDICES == i).any(axis=1))[0] for i in range(N)]
# Also precompute for each triple row, the other two points
# For triple (i,j,k), if we move point i, we need j and k
# triple_other_points[row] = (other1, other2) relative to which point
# Actually, let's precompute per-point: for point p in triple t=(a,b,c), the other two are the remaining two
_POINT_OTHERS = []  # _POINT_OTHERS[p] = list of (triple_row, other1, other2)
for p in range(N):
    rows = _TRIPLE_POINT_TO_IDX[p]
    others = []
    for r in rows:
        a, b, c = TRIPLE_INDICES[r]
        if p == a:
            others.append((r, b, c))
        elif p == b:
            others.append((r, a, c))
        else:
            others.append((r, a, b))
    _POINT_OTHERS.append(others)

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

def _optimize_ring(n_out, n_in, time_limit, start_time=0, deadline=0):
    """
    Optimize a specific ring configuration.
    """
    if start_time == 0:
        start_time = time.time()
    if deadline == 0:
        deadline = start_time + time_limit

    g = np.gcd(n_out, n_in)
    theta_max = 2 * np.pi / g
    
    bounds = [(0.0, 0.999), (0.0, theta_max)]
    
    best_score = 0.0
    best_params = None
    
    # Use Differential Evolution
    # We want to be efficient.
    # If time_limit is large, we can use more iterations.
    
    # Let's try to be a bit more aggressive with popsize if time permits
    # But we need to be careful with the total time budget.
    # For 15s, maybe 20-30 iterations?
    
    popsize = 10
    maxiter = 30
    
    # If we have more time, increase popsize and maxiter
    if time_limit > 30:
        popsize = 15
        maxiter = 50
    elif time_limit > 15:
        popsize = 12
        maxiter = 40

    try:
        res = differential_evolution(
            lambda p: -_score_ring_params(n_out, n_in, p),
            bounds,
            seed=42,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-5,
            polish=True,
            # We don't have a strict time callback for DE in older scipy,
            # but we can rely on the maxiter/popsize limits being conservative enough
            # or we can implement a timeout wrapper if needed.
            # Given the constraint, let's assume 30 iters * 10 pop * eval time < time_limit
        )
        
        if res.success:
            best_score = -res.fun
            best_params = res.x
    except Exception as e:
        pass
    
    # If we have time, do a local polish with multiple starts
    if time.time() < deadline - 5.0:
        starts = []
        # Add some heuristic starts
        # r around 0.5, 0.7, 0.9
        # theta 0, theta_max/2
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

def _fast_soft_min_score(pts, beta=SOFT_BETA):
    """
    Fast soft-min score that uses cached triple areas where possible.
    For L-BFGS-B, we recompute all areas but in a vectorized way.
    """
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    
    p_i = pts[TRIPLE_INDICES[:, 0]]
    p_j = pts[TRIPLE_INDICES[:, 1]]
    p_k = pts[TRIPLE_INDICES[:, 2]]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    
    areas = 0.5 * np.abs(cross)
    
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    soft_min_val = (1.0 / beta) * (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled))))
    
    return soft_min_val / hull_area


def _local_update_polish(points, deadline):
    """
    Local-update coordinate descent: move one point at a time, only recomputing
    affected triple areas. Much faster than full recompute per evaluation.
    """
    pts = points.copy()
    score = _calculate_score(pts)
    if score <= 0:
        return pts, score
    
    # Precompute all triple areas
    p_i = pts[TRIPLE_INDICES[:, 0]]
    p_j = pts[TRIPLE_INDICES[:, 1]]
    p_k = pts[TRIPLE_INDICES[:, 2]]
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    areas = 0.5 * np.abs(cross)
    
    # Compute hull area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return pts, score
    
    if hull_area <= 1e-9:
        return pts, score
    
    current_min = np.min(areas)
    current_ratio = current_min / hull_area
    
    # For each point, try small perturbations
    n_iter = 0
    max_iter = 200
    step_size = 0.01
    
    for iteration in range(max_iter):
        if time.time() > deadline - 2.0:
            break
        
        improved = False
        for p in range(N):
            if time.time() > deadline - 2.0:
                break
            
            # Get affected triples
            affected_rows = _TRIPLE_POINT_TO_IDX[p]
            
            # Try 8 directions + 4 step sizes
            best_local_pts = pts[p].copy()
            best_local_ratio = current_ratio
            
            for dx in [-step_size, 0, step_size]:
                for dy in [-step_size, 0, step_size]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    new_px = pts[p, 0] + dx
                    new_py = pts[p, 1] + dy
                    
                    # Recompute only affected areas
                    new_areas = areas.copy()
                    for r, other1, other2 in _POINT_OTHERS[p]:
                        pi = pts[TRIPLE_INDICES[r, 0]]
                        pj = pts[TRIPLE_INDICES[r, 1]]
                        pk = pts[TRIPLE_INDICES[r, 2]]
                        
                        # Replace point p in the triple
                        if p == TRIPLE_INDICES[r, 0]:
                            c = (pj[0] - new_px) * (pk[1] - new_py) - (pj[1] - new_py) * (pk[0] - new_px)
                        elif p == TRIPLE_INDICES[r, 1]:
                            c = (pi[0] - new_px) * (pk[1] - new_py) - (pi[1] - new_py) * (pk[0] - new_px)
                        else:
                            c = (pi[0] - new_px) * (pj[1] - new_py) - (pi[1] - new_py) * (pj[0] - new_px)
                        new_areas[r] = 0.5 * abs(c)
                    
                    new_min = np.min(new_areas)
                    # Approximate hull area change (small perturbation)
                    new_ratio = new_min / hull_area
                    
                    if new_ratio > best_local_ratio + 1e-10:
                        best_local_ratio = new_ratio
                        best_local_pts[0] = new_px
                        best_local_pts[1] = new_py
            
            if not np.allclose(best_local_pts, pts[p]):
                pts[p] = best_local_pts
                # Update affected areas
                for r, other1, other2 in _POINT_OTHERS[p]:
                    pi = pts[TRIPLE_INDICES[r, 0]]
                    pj = pts[TRIPLE_INDICES[r, 1]]
                    pk = pts[TRIPLE_INDICES[r, 2]]
                    c = (pj[0] - pi[0]) * (pk[1] - pi[1]) - (pj[1] - pi[1]) * (pk[0] - pi[0])
                    areas[r] = 0.5 * abs(c)
                
                current_min = np.min(areas)
                current_ratio = current_min / hull_area
                improved = True
        
        if not improved:
            # Try smaller step
            step_size *= 0.5
            if step_size < 1e-5:
                break
    
    final_score = _calculate_score(pts)
    return pts, final_score


def _coordinate_polish(points, n_out, n_in, time_budget, deadline):
    """
    Polish all 28 coordinates using multi-start L-BFGS-B with soft-min surrogate,
    plus local-update coordinate descent for fine-tuning.
    Uses beta annealing to improve convergence.
    """
    if time.time() > deadline - 3.0:
        return points, _calculate_score(points)
    
    best_points = points.copy()
    best_score = _calculate_score(points)
    
    # Beta Annealing Schedule
    betas = [10.0, 50.0, 200.0]
    
    for beta in betas:
        if time.time() > deadline - 5.0:
            break
            
        def neg_soft_score(x):
            pts = x.reshape(-1, 2)
            return _fast_soft_min_score(pts, beta=beta)
        
        # Main optimization step
        try:
            res = minimize(
                neg_soft_score,
                best_points.flatten(),
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
        
        # Multi-start for lower betas to escape local minima
        if beta < 100.0 and time.time() < deadline - 10.0:
            np.random.seed(42)
            for trial in range(3):
                if time.time() > deadline - 10.0:
                    break
                
                perturbation = np.random.normal(0, 0.05, (N, 2))
                start_pts = best_points + perturbation
                
                try:
                    res_t = minimize(
                        neg_soft_score,
                        start_pts.flatten(),
                        method='L-BFGS-B',
                        options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                    )
                    
                    polished_t = res_t.x.reshape(-1, 2)
                    polished_t_score = _calculate_score(polished_t)
                    
                    if polished_t_score > best_score:
                        best_points = polished_t
                        best_score = polished_t_score
                        
                except Exception:
                    pass

    # Final Local Update Polish (Coordinate Descent)
    if time.time() < deadline - 2.0:
        local_deadline = deadline - 2.0
        best_points, best_score = _local_update_polish(best_points, local_deadline)
        
    return best_points, best_score

def _optimize_2p(n_out, n_in, time_limit, start_time, deadline):
    """
    Optimize the 2-parameter ring configuration (r, theta).
    """
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

    try:
        res = differential_evolution(
            lambda p: _soft_min_ring_score(n_out, n_in, p),
            bounds,
            seed=42,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-5,
            polish=True
        )
        
        if res.success:
            best_params = res.x
            r, theta = best_params
            best_points = _get_points_2p(n_out, n_in, r, theta)
            best_score = _calculate_score(best_points)
    except Exception as e:
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
        
    # Return the actual polished points as a flat array for the caller to reshape
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
    Uses L-BFGS-B with beta annealing for better convergence.
    """
    start_time = time.time()
    
    # Initial parameters
    # M = Identity
    # delta = 0
    best_params = np.zeros(20)
    best_params[0] = 1.0
    best_params[4] = 1.0
    
    best_points = _generate_jittered_points(best_params)
    best_score = _calculate_score(best_points)
    
    # Beta Annealing Schedule
    betas = [10.0, 50.0, 200.0]
    
    for beta in betas:
        if time.time() > deadline - 5.0:
            break
            
        def neg_soft_obj(p):
            return _soft_min_jittered_score(p, beta=beta)
        
        # Main optimization step
        try:
            res = minimize(
                neg_soft_obj,
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if -res.fun > best_score:
                best_params = res.x
                best_points = _generate_jittered_points(best_params)
                best_score = _calculate_score(best_points)
                
        except Exception:
            pass
            
        # Random restarts for lower betas
        if beta < 100.0 and time.time() < deadline - 10.0:
            np.random.seed(123)
            for trial in range(3):
                if time.time() > deadline - 10.0:
                    break
                    
                p_init = np.zeros(20)
                p_init[0] = 1.0
                p_init[4] = 1.0
                p_init[1] = np.random.uniform(-0.1, 0.1)
                p_init[3] = np.random.uniform(-0.1, 0.1)
                p_init[6:] = np.random.uniform(-0.05, 0.05, 14)
                
                try:
                    res_t = minimize(
                        neg_soft_obj,
                        p_init,
                        method='L-BFGS-B',
                        options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                    )
                    
                    if -res_t.fun > best_score:
                        best_params = res_t.x
                        best_points = _generate_jittered_points(best_params)
                        best_score = _calculate_score(best_points)
                        
                except Exception:
                    pass

    # Final coordinate-level polish on the best points
    if time.time() < deadline - 3.0:
        coord_deadline = deadline - 3.0
        best_points, best_score = _coordinate_polish(best_points, 7, 7, 10.0, coord_deadline)
            
    return best_score, best_points

# NEW STRUCTURAL APPROACH: FOURIER CURVATURE MODEL
# Represents 14 points as vertices of a convex polygon defined by r(theta) = 1 + sum c_k cos(k*theta)
# This allows asymmetric "breathing" shapes while maintaining convexity (if coefficients are small enough)
# Reduces search space to 3 parameters (c1, c2, c3) + potentially rotation/scale, but here we fix scale=1 and rotation=0
# and optimize c1, c2, c3. Then we do coordinate polish.

def _generate_fourier_points(params):
    """
    Generate 14 points from Fourier curvature model.
    params: [c1, c2, c3]
    r(theta) = 1 + c1*cos(theta) + c2*cos(2*theta) + c3*cos(3*theta)
    Points are at angles theta_i = 2*pi*i/14
    """
    c1, c2, c3 = params
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    
    # Compute r(theta) for each angle
    r = 1.0 + c1 * np.cos(angles) + c2 * np.cos(2*angles) + c3 * np.cos(3*angles)
    
    # Ensure r > 0 for convexity (if r becomes negative or zero, points collapse or cross)
    # We'll handle this in the objective function by penalizing or returning 0
    # For now, just generate points
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    
    return np.column_stack((x, y))

def _soft_min_fourier_score(params, beta=SOFT_BETA):
    """
    Soft-min score for Fourier model.
    """
    # Check if r(theta) is positive everywhere
    c1, c2, c3 = params
    # Sample many points to check positivity
    theta_check = np.linspace(0, 2*np.pi, 100)
    r_check = 1.0 + c1 * np.cos(theta_check) + c2 * np.cos(2*theta_check) + c3 * np.cos(3*theta_check)
    if np.min(r_check) <= 0.01:  # Small margin for safety
        return 0.0  # Invalid, non-convex or degenerate
    
    points = _generate_fourier_points(params)
    return _soft_min_score(points, beta=beta)

def _optimize_fourier(time_limit, deadline):
    """
    Optimize the Fourier curvature model.
    Uses differential evolution for global search, then L-BFGS-B for local polish,
    then coordinate-level polish.
    """
    start_time = time.time()
    
    # Bounds for c1, c2, c3
    # To maintain convexity, we need |c1| + |c2| + |c3| < 1 (roughly)
    # Let's use bounds [-0.4, 0.4] for each
    bounds = [(-0.4, 0.4), (-0.4, 0.4), (-0.4, 0.4)]
    
    best_score = 0.0
    best_params = None
    best_points = None
    
    # Phase 1: Differential Evolution
    try:
        res = differential_evolution(
            lambda p: _soft_min_fourier_score(p),
            bounds,
            seed=42,
            maxiter=30,
            popsize=10,
            tol=1e-5,
            polish=True
        )
        
        if res.success:
            best_params = res.x
            best_points = _generate_fourier_points(best_params)
            best_score = _calculate_score(best_points)
    except Exception as e:
        pass
    
    # Phase 2: Local polish with L-BFGS-B
    if best_params is not None and time.time() < deadline - 5.0:
        try:
            def neg_soft_fourier(p):
                return _soft_min_fourier_score(p, beta=200.0)
            
            res_nm = minimize(
                neg_soft_fourier,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if -res_nm.fun > best_score:
                best_params = res_nm.x
                best_points = _generate_fourier_points(best_params)
                best_score = _calculate_score(best_points)
        except Exception:
            pass
    
    # Phase 3: Coordinate-level polish to escape Fourier parameterization
    if best_points is not None and time.time() < deadline - 3.0:
        coord_deadline = deadline - 3.0
        best_points, best_score = _coordinate_polish(best_points, 7, 7, 10.0, coord_deadline)
    
    if best_points is None:
        return 0.0, None
    
    return best_score, best_points

def _optimize_double_star(time_limit, deadline):
    """
    Optimize the 7-7 ring with theta fixed to pi/7 (double-star topology).
    This reduces the problem to 1 parameter: r.
    """
    if time.time() > deadline - 5.0:
        return 0.0, None
    
    n_out, n_in = 7, 7
    theta_fixed = np.pi / 7.0
    
    # Define objective for 1D optimization
    def neg_score_1d(r):
        if r < 0 or r > 0.999:
            return 0.0
        points = _get_points_2p(n_out, n_in, r, theta_fixed)
        return -_calculate_score(points)
    
    best_score = 0.0
    best_r = 0.5
    
    # Phase 1: Coarse scan
    r_vals = np.linspace(0.1, 0.99, 20)
    for r in r_vals:
        s = -neg_score_1d(r)
        if s > best_score:
            best_score = s
            best_r = r
            
    # Phase 2: Fine polish around best_r
    if time.time() < deadline - 5.0:
        r_min = max(0.1, best_r - 0.1)
        r_max = min(0.99, best_r + 0.1)
        
        try:
            res = minimize_scalar(
                neg_score_1d,
                bounds=(r_min, r_max),
                method='bounded',
                options={'xatol': 1e-8}
            )
            if res.success:
                best_r = res.x
                best_score = -res.fun
        except Exception:
            pass
            
    best_points = _get_points_2p(n_out, n_in, best_r, theta_fixed)
    
    # Phase 3: Coordinate polish to break the strict double-star symmetry if beneficial
    if time.time() < deadline - 5.0:
        coord_deadline = deadline - 5.0
        best_points, best_score = _coordinate_polish(best_points, n_out, n_in, 10.0, coord_deadline)
        
    return best_score, best_points

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    candidates = [
        (7, 7),
        (8, 6),
        (6, 8),
    ]
    
    best_overall_score = 0.0
    best_overall_points = None
    
    # Strategy 1: Double Star (1-param 7-7 ring)
    # This is very fast and likely close to optimal.
    ds_deadline = start_time + 20.0
    try:
        ds_score, ds_points = _optimize_double_star(20.0, ds_deadline)
        if ds_score > best_overall_score:
            best_overall_score = ds_score
            best_overall_points = ds_points.copy()
    except Exception:
        pass
    
    # Strategy 2: General 7-7 Ring (2-param)
    # Allow theta to vary slightly from pi/7
    ring_deadline = start_time + 50.0
    try:
        score, params = _optimize_2p(7, 7, 30.0, start_time, ring_deadline)
        if score > best_overall_score:
            best_overall_score = score
            if params is not None:
                best_overall_points = params.reshape(-1, 2).copy()
                actual_score = _calculate_score(best_overall_points)
                best_overall_score = actual_score
    except Exception:
        pass
        
    # Strategy 3: Other rings if time permits
    current_time = time.time()
    if current_time < deadline - 10:
        for n_out, n_in in [(8, 6), (6, 8)]:
            if time.time() > deadline - 10:
                break
            t_limit = 10.0
            try:
                score, params = _optimize_2p(n_out, n_in, t_limit, start_time, deadline)
                if score > best_overall_score:
                    best_overall_score = score
                    if params is not None:
                        best_overall_points = params.reshape(-1, 2).copy()
                        actual_score = _calculate_score(best_overall_points)
                        best_overall_score = actual_score
            except Exception:
                pass

    # Final Polish on the best candidate
    if best_overall_points is not None and time.time() < deadline - 5:
        # Use remaining time for a final coordinate polish
        coord_deadline = deadline - 2.0
        best_overall_points, best_overall_score = _coordinate_polish(best_overall_points, 7, 7, 10.0, coord_deadline)
        
    # Fallback
    if best_overall_points is None:
        n_out, n_in = 7, 7
        r, theta = 0.5, 0.0
        best_overall_points = _get_points_2p(n_out, n_in, r, theta)
        
    return best_overall_points
