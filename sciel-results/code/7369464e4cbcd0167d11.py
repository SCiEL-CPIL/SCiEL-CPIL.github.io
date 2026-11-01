import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14

def _get_triangle_indices(n):
    """Pre-compute all (i,j,k) with i<j<k"""
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=np.int32)

_TRI_INDICES = _get_triangle_indices(N)
_I = _TRI_INDICES[:, 0]
_J = _TRI_INDICES[:, 1]
_K = _TRI_INDICES[:, 2]

def _min_triangle_area(points):
    """Compute minimum triangle area for given points."""
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross).min()

def _soft_min_triangle_area(points, beta=100.0):
    """
    Smooth surrogate for min triangle area using log-sum-exp.
    Numerically stable version.
    """
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    # Soft min via log-sum-exp (numerically stable)
    max_a = np.max(areas)
    shifted = areas - max_a
    log_sum = np.log(np.sum(np.exp(-beta * shifted))) / beta
    soft_min = max_a - log_sum
    return soft_min

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = scipy.spatial.ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Score = min triangle area / convex hull area."""
    try:
        if len(points) < 3:
            return 0.0
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
        mta = _min_triangle_area(points)
        return mta / ha
    except Exception:
        return 0.0

def _neg_score(points_flat):
    """Negative score for minimization."""
    points = points_flat.reshape(-1, 2)
    return -_score(points)

def _neg_soft_score(points_flat, beta=100.0):
    """Negative soft-score for gradient-based optimization."""
    points = points_flat.reshape(-1, 2)
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
        smta = _soft_min_triangle_area(points, beta)
        return -(smta / ha)
    except Exception:
        return 0.0

def _normalize(points):
    """Scale so convex hull area = 1."""
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return points
        return points / np.sqrt(ha)
    except Exception:
        return points

def _regular_polygon(n, radius=1.0, rotation=0.0):
    """Regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _generate_parameterized_points(r, delta):
    """
    Generate 14 points: 7 on outer circle (radius 1) and 7 on inner circle (radius r).
    Outer points: angles 0, 2pi/7, ...
    Inner points: angles delta, delta + 2pi/7, ...
    """
    n_outer = 7
    n_inner = 7
    
    angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    points_outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    
    angles_inner = np.linspace(delta, delta + 2 * np.pi, n_inner, endpoint=False)
    points_inner = np.column_stack([r * np.cos(angles_inner), r * np.sin(angles_inner)])
    
    return np.vstack([points_outer, points_inner])

def _score_parameterized(r, delta):
    """Score the parameterized configuration."""
    try:
        if r <= 0:
            return 0.0
        pts = _generate_parameterized_points(r, delta)
        pts_norm = _normalize(pts)
        return _score(pts_norm)
    except Exception:
        return 0.0

def _neg_score_parameterized(params):
    """Negative score for parameterized optimization."""
    r, delta = params
    if r <= 0:
        r = 1e-6
    return -_score_parameterized(r, delta)

def _generate_flexible_ring_points(r_inner, delta_inner, angle_jitter_outer, angle_jitter_inner):
    """
    Generate 14 points with flexible angular positions on two rings.
    Outer ring: radius 1, base angles 0, 2pi/7, ... + jitter
    Inner ring: radius r_inner, base angles delta, delta+2pi/7, ... + jitter
    """
    n_ring = 7
    base_angles = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
    
    outer_angles = base_angles + angle_jitter_outer
    inner_angles = base_angles + delta_inner + angle_jitter_inner
    
    points_outer = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    points_inner = np.column_stack([r_inner * np.cos(inner_angles), r_inner * np.sin(inner_angles)])
    
    return np.vstack([points_outer, points_inner])

def _neg_score_flexible(params):
    """
    Negative score for flexible ring parameterization.
    params: [r_inner, delta_inner, *outer_jitter(7), *inner_jitter(7)]
    Total: 16 parameters
    """
    try:
        r_inner = params[0]
        delta_inner = params[1]
        outer_jitter = params[2:9]
        inner_jitter = params[9:16]
        
        if r_inner <= 0:
            r_inner = 1e-6
        
        pts = _generate_flexible_ring_points(r_inner, delta_inner, outer_jitter, inner_jitter)
        pts_norm = _normalize(pts)
        return -_score(pts_norm)
    except Exception:
        return 0.0

def _neg_soft_score_flexible(params, beta=50.0):
    """
    Negative soft-score for flexible ring parameterization (for L-BFGS-B).
    """
    try:
        r_inner = params[0]
        delta_inner = params[1]
        outer_jitter = params[2:9]
        inner_jitter = params[9:16]
        
        if r_inner <= 0:
            r_inner = 1e-6
        
        pts = _generate_flexible_ring_points(r_inner, delta_inner, outer_jitter, inner_jitter)
        ha = _hull_area(pts)
        if ha < 1e-12:
            return 0.0
        smta = _soft_min_triangle_area(pts, beta)
        return -(smta / ha)
    except Exception:
        return 0.0

def _regular_14_gon():
    return _regular_polygon(14, 1.0, 0.0)

def _single_center_13_around():
    """1 point at center, 13 on a circle."""
    center = np.array([[0.0, 0.0]])
    outer = _regular_polygon(13, 1.0, 0.0)
    return np.vstack([center, outer])

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_score = 0.0
    best_points = None
    
    def _update_best(pts, sc):
        nonlocal best_score, best_points
        if sc > best_score:
            best_score = sc
            best_points = _normalize(pts).copy()

    def _elapsed():
        return time.time() - start_time

    def _time_left():
        return deadline - time.time()

    # Phase 1: Coarse grid search on 2-param parameterization
    r_vals = np.linspace(0.1, 0.95, 20)
    delta_vals = np.linspace(0, 2 * np.pi / 7, 15, endpoint=False)
    
    grid_scores = np.zeros((len(r_vals), len(delta_vals)))
    
    for i, r in enumerate(r_vals):
        for j, delta in enumerate(delta_vals):
            sc = _score_parameterized(r, delta)
            grid_scores[i, j] = sc
            if sc > best_score:
                update_pts = _generate_parameterized_points(r, delta)
                _update_best(update_pts, sc)
    
    if _time_left() < 20:
        return best_points if best_points is not None else _normalize(_regular_14_gon())

    # Find best grid point
    idx_max = np.unravel_index(np.argmax(grid_scores), grid_scores.shape)
    r0 = r_vals[idx_max[0]]
    delta0 = delta_vals[idx_max[1]]
    
    # Phase 2: Refine 2-param with Nelder-Mead
    try:
        res = scipy.optimize.minimize(
            _neg_score_parameterized,
            [r0, delta0],
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10}
        )
        r_opt, delta_opt = res.x
        if r_opt > 0:
            pts_opt = _generate_parameterized_points(r_opt, delta_opt)
            sc_opt = _score(_normalize(pts_opt))
            _update_best(pts_opt, sc_opt)
    except Exception:
        pass

    if _time_left() < 15:
        return best_points if best_points is not None else _normalize(_regular_14_gon())

    # Phase 3: Flexible ring optimization (16 params)
    if best_points is not None:
        init_flex = np.array([r_opt, delta_opt] + [0.0]*14)
    else:
        init_flex = np.array([0.5, 0.0] + [0.0]*14)

    # First pass: Nelder-Mead on flexible params
    try:
        res_flex = scipy.optimize.minimize(
            _neg_score_flexible,
            init_flex,
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        pts_flex = _generate_flexible_ring_points(
            max(res_flex.x[0], 1e-6), res_flex.x[1], 
            res_flex.x[2:9], res_flex.x[9:16]
        )
        sc_flex = _score(_normalize(pts_flex))
        _update_best(pts_flex, sc_flex)
    except Exception:
        pass

    if _time_left() < 10:
        return best_points if best_points is not None else _normalize(_regular_14_gon())

    # Phase 4: Progressive L-BFGS-B with soft surrogate for high-precision polish
    # Start with low beta (smooth) and increase to high beta (sharp)
    current_x = res_flex.x if 'res_flex' in dir() else init_flex
    for beta in [20.0, 50.0, 100.0, 200.0]:
        if _time_left() < 5:
            break
        try:
            res_lbfgs = scipy.optimize.minimize(
                lambda x, b=beta: _neg_soft_score_flexible(x, b),
                current_x,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            pts_lbfgs = _generate_flexible_ring_points(
                max(res_lbfgs.x[0], 1e-6), res_lbfgs.x[1],
                res_lbfgs.x[2:9], res_lbfgs.x[9:16]
            )
            sc_lbfgs = _score(_normalize(pts_lbfgs))
            _update_best(pts_lbfgs, sc_lbfgs)
            current_x = res_lbfgs.x
        except Exception:
            pass

    if _time_left() < 8:
        return best_points if best_points is not None else _normalize(_regular_14_gon())

    # Phase 5: Full 28D local optimization from best point using soft surrogate
    if best_points is not None:
        try:
            # Use soft surrogate for gradient-based full-coordinate optimization
            for beta in [50.0, 100.0]:
                if _time_left() < 5:
                    break
                res_full = scipy.optimize.minimize(
                    lambda x, b=beta: _neg_soft_score(x, b),
                    best_points.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                pts_full = _normalize(res_full.x.reshape(-1, 2))
                sc_full = _score(pts_full)
                _update_best(pts_full, sc_full)
        except Exception:
            pass

    if _time_left() < 5:
        return best_points if best_points is not None else _normalize(_regular_14_gon())

    # Phase 6: Random restarts with flexible ring structure
    rng = np.random.RandomState(42)
    
    for _ in range(20):
        if _time_left() < 5:
            break
        
        # Generate flexible ring init
        r_rand = rng.uniform(0.2, 0.9)
        delta_rand = rng.uniform(0, 2 * np.pi / 7)
        jitter_o = rng.normal(0, 0.1, 7)
        jitter_i = rng.normal(0, 0.1, 7)
        
        init_rand = np.concatenate([[r_rand, delta_rand], jitter_o, jitter_i])
        
        try:
            res_rand = scipy.optimize.minimize(
                _neg_score_flexible,
                init_rand,
                method='Nelder-Mead',
                options={'maxiter': 1500, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            pts_rand = _generate_flexible_ring_points(
                max(res_rand.x[0], 1e-6), res_rand.x[1],
                res_rand.x[2:9], res_rand.x[9:16]
            )
            sc_rand = _score(_normalize(pts_rand))
            _update_best(pts_rand, sc_rand)
        except Exception:
            pass

    # Phase 7: Final high-precision polish with L-BFGS-B on full 28D using soft surrogate
    if best_points is not None and _time_left() > 3:
        try:
            for beta in [100.0, 200.0, 500.0]:
                if _time_left() < 3:
                    break
                res_polish = scipy.optimize.minimize(
                    lambda x, b=beta: _neg_soft_score(x, b),
                    best_points.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                pts_polish = _normalize(res_polish.x.reshape(-1, 2))
                sc_polish = _score(pts_polish)
                _update_best(pts_polish, sc_polish)
        except Exception:
            pass

    if best_points is None:
        best_points = _normalize(_regular_14_gon())
    else:
        best_points = _normalize(best_points)
        
    return best_points
