import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14

def _get_triangle_indices(n):
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
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross).min()

def _hull_area(points):
    try:
        hull = scipy.spatial.ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
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
    points = points_flat.reshape(-1, 2)
    return -_score(points)

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
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _generate_ring_config(radii, angles):
    """
    Generate points from ring configuration.
    radii: list of radii per point (length N)
    angles: list of angles per point (length N)
    """
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack([x, y])

def _two_ring_7_7(r, delta):
    """7 outer (radius 1), 7 inner (radius r), offset by delta."""
    angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    angles_inner = np.linspace(delta, delta + 2 * np.pi, 7, endpoint=False)
    all_angles = np.concatenate([angles_outer, angles_inner])
    all_radii = np.concatenate([np.ones(7), r * np.ones(7)])
    return _generate_ring_config(all_radii, all_angles)

def _three_ring(n1, n2, n3, r1, r2, r3, rot1, rot2, rot3):
    """Three concentric rings."""
    a1 = np.linspace(rot1, rot1 + 2 * np.pi, n1, endpoint=False)
    a2 = np.linspace(rot2, rot2 + 2 * np.pi, n2, endpoint=False)
    a3 = np.linspace(rot3, rot3 + 2 * np.pi, n3, endpoint=False)
    all_angles = np.concatenate([a1, a2, a3])
    all_radii = np.concatenate([r1 * np.ones(n1), r2 * np.ones(n2), r3 * np.ones(n3)])
    return _generate_ring_config(all_radii, all_angles)

def _four_ring(n1, n2, n3, n4, r1, r2, r3, r4, rot1, rot2, rot3, rot4):
    """Four concentric rings."""
    a1 = np.linspace(rot1, rot1 + 2 * np.pi, n1, endpoint=False)
    a2 = np.linspace(rot2, rot2 + 2 * np.pi, n2, endpoint=False)
    a3 = np.linspace(rot3, rot3 + 2 * np.pi, n3, endpoint=False)
    a4 = np.linspace(rot4, rot4 + 2 * np.pi, n4, endpoint=False)
    all_angles = np.concatenate([a1, a2, a3, a4])
    all_radii = np.concatenate([r1 * np.ones(n1), r2 * np.ones(n2), r3 * np.ones(n3), r4 * np.ones(n4)])
    return _generate_ring_config(all_radii, all_angles)

def _neg_score_two_ring(params):
    r, delta = params
    if r <= 0:
        r = 1e-6
    pts = _two_ring_7_7(r, delta)
    return -_score(_normalize(pts))

def _neg_score_three_ring_752(params):
    r1, r2, r3, rot1, rot2, rot3 = params
    r1, r2, r3 = max(r1, 1e-6), max(r2, 1e-6), max(r3, 1e-6)
    pts = _three_ring(7, 5, 2, r1, r2, r3, rot1, rot2, rot3)
    return -_score(_normalize(pts))

def _neg_score_three_ring_662(params):
    r1, r2, r3, rot1, rot2, rot3 = params
    r1, r2, r3 = max(r1, 1e-6), max(r2, 1e-6), max(r3, 1e-6)
    pts = _three_ring(6, 6, 2, r1, r2, r3, rot1, rot2, rot3)
    return -_score(_normalize(pts))

def _neg_score_three_ring_554(params):
    r1, r2, r3, rot1, rot2, rot3 = params
    r1, r2, r3 = max(r1, 1e-6), max(r2, 1e-6), max(r3, 1e-6)
    pts = _three_ring(5, 5, 4, r1, r2, r3, rot1, rot2, rot3)
    return -_score(_normalize(pts))

def _neg_score_four_ring_4433(params):
    r1, r2, r3, r4, rot1, rot2, rot3, rot4 = params
    r1, r2, r3, r4 = max(r1, 1e-6), max(r2, 1e-6), max(r3, 1e-6), max(r4, 1e-6)
    pts = _four_ring(4, 4, 3, 3, r1, r2, r3, r4, rot1, rot2, rot3, rot4)
    return -_score(_normalize(pts))

def _neg_score_flexible(params):
    """
    Flexible parameterization: each point has radius and angle.
    params: 28 values [r0, a0, r1, a1, ..., r13, a13]
    """
    radii = params[0::2]
    angles = params[1::2]
    radii = np.maximum(radii, 1e-6)
    pts = _generate_ring_config(radii, angles)
    return -_score(_normalize(pts))

def _neg_soft_score_flexible(params, beta=100.0):
    """
    Soft surrogate for flexible parameterization.
    Uses log-sum-exp for smooth min.
    """
    radii = params[0::2]
    angles = params[1::2]
    radii = np.maximum(radii, 1e-6)
    pts = _generate_ring_config(radii, angles)
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 0.0
    pi = pts[_I]
    pj = pts[_J]
    pk = pts[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    max_a = np.max(areas)
    if max_a < 1e-15:
        return 0.0
    log_sum = np.log(np.sum(np.exp(-beta * (areas - max_a)))) / beta
    soft_min = max_a - log_sum
    return -(soft_min / ha)

def _neg_smooth_score_and_grad(points_flat, beta=100.0):
    """
    Smooth surrogate for -score using log-sum-exp soft-min of triangle areas.
    Returns (value, gradient) for L-BFGS-B.
    """
    points = points_flat.reshape(-1, 2)
    
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    max_area = np.max(areas)
    if max_area < 1e-15:
        return 0.0, np.zeros_like(points_flat)
    
    # Soft min via log-sum-exp
    scaled = -beta * (areas - max_area)
    scaled = np.clip(scaled, -700, 700)
    log_sum = np.log(np.sum(np.exp(scaled)))
    soft_min = max_area - log_sum / beta
    
    # Compute gradient of soft_min w.r.t. points
    weights = np.exp(scaled - log_sum)  # softmax weights
    weights = weights / beta
    
    # Gradient of cross w.r.t. each point
    # cross_i = (p_j - p_i) x (p_k - p_i)
    # d(cross)/d(p_i) = -(v1 + v2) rotated 90deg... actually let's use vector calculus
    # cross = (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i)
    # d(cross)/dx_i = -(y_k - y_i) + (y_j - y_i) = y_j - y_k
    # d(cross)/dy_i = (x_k - x_i) - (x_j - x_i) = x_k - x_j
    # d(cross)/dx_j = y_k - y_i
    # d(cross)/dy_j = -(x_k - x_i)
    # d(cross)/dx_k = -(y_j - y_i)
    # d(cross)/dy_k = x_j - x_i
    
    grad_points = np.zeros_like(points)
    
    # For each triangle, accumulate gradients
    for idx in range(len(_I)):
        i, j, k = _I[idx], _J[idx], _K[idx]
        w = weights[idx]
        a = areas[idx]
        # sign of cross determines sign of gradient of |cross|
        sgn = 1.0 if cross[idx] >= 0 else -1.0
        w_eff = w * sgn
        
        # d(cross)/d(p_i)
        grad_points[i, 0] += -w_eff * (pj[idx, 1] - pk[idx, 1])
        grad_points[i, 1] += -w_eff * (pk[idx, 0] - pj[idx, 0])
        
        # d(cross)/d(p_j)
        grad_points[j, 0] += -w_eff * (pk[idx, 1] - pi[idx, 1])
        grad_points[j, 1] += -w_eff * (pi[idx, 0] - pk[idx, 0])
        
        # d(cross)/d(p_k)
        grad_points[k, 0] += -w_eff * (pi[idx, 1] - pj[idx, 1])
        grad_points[k, 1] += -w_eff * (pj[idx, 0] - pi[idx, 0])
    
    # d(soft_min)/d(points) = 0.5 * grad_points
    # d(-soft_min/ha)/d(points) = -0.5 * grad_points / ha
    ha = _hull_area(points)
    if ha < 1e-12:
        return 0.0, np.zeros_like(points_flat)
    
    value = -(soft_min / ha)
    gradient = -0.5 * grad_points / ha
    
    return value, gradient.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    best_score = 0.0
    best_points = None

    def _update_best(pts, sc):
        nonlocal best_score, best_points
        if sc > best_score:
            best_score = sc
            best_points = _normalize(pts).copy()

    def _time_left():
        return deadline - time.time()

    # Start with regular 14-gon as baseline
    reg14 = _regular_polygon(14, 1.0, 0.0)
    sc_reg = _score(_normalize(reg14))
    _update_best(reg14, sc_reg)

    # Phase 1: Coarse grid search on two-ring 7/7
    r_vals = np.linspace(0.1, 0.95, 25)
    delta_vals = np.linspace(0, 2 * np.pi / 7, 20, endpoint=False)

    best_two_ring = None
    best_two_ring_score = 0.0

    for r in r_vals:
        for delta in delta_vals:
            sc = -_neg_score_two_ring([r, delta])
            if sc > best_two_ring_score:
                best_two_ring_score = sc
                best_two_ring = [r, delta]

    if best_two_ring is not None:
        pts_tr = _two_ring_7_7(best_two_ring[0], best_two_ring[1])
        _update_best(pts_tr, best_two_ring_score)

    if _time_left() < 30:
        return best_points if best_points is not None else _normalize(reg14)

    # Phase 2: Refine two-ring with Nelder-Mead
    try:
        res = scipy.optimize.minimize(
            _neg_score_two_ring,
            best_two_ring,
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
        )
        pts_opt = _two_ring_7_7(res.x[0], res.x[1])
        sc_opt = _score(_normalize(pts_opt))
        _update_best(pts_opt, sc_opt)
    except Exception:
        pass

    if _time_left() < 25:
        return best_points if best_points is not None else _normalize(reg14)

    # Phase 3: Three-ring configurations with differential evolution
    configs = [
        (7, 5, 2, _neg_score_three_ring_752, "752"),
        (6, 6, 2, _neg_score_three_ring_662, "662"),
        (5, 5, 4, _neg_score_three_ring_554, "554"),
    ]

    bounds_3 = [(0.05, 1.5), (0.05, 1.5), (0.05, 1.5),
                (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi)]

    for n1, n2, n3, func, name in configs:
        if _time_left() < 15:
            break
        try:
            # Multiple starts for DE
            for seed in [42, 123, 456]:
                if _time_left() < 10:
                    break
                de_res = scipy.optimize.differential_evolution(
                    func,
                    bounds=bounds_3,
                    seed=seed,
                    maxiter=80,
                    tol=1e-8,
                    polish=True,
                    popsize=15
                )
                if name == "752":
                    pts_de = _three_ring(7, 5, 2, *de_res.x)
                elif name == "662":
                    pts_de = _three_ring(6, 6, 2, *de_res.x)
                else:
                    pts_de = _three_ring(5, 5, 4, *de_res.x)
                sc_de = _score(_normalize(pts_de))
                _update_best(pts_de, sc_de)
        except Exception:
            pass

    if _time_left() < 15:
        return best_points if best_points is not None else _normalize(reg14)

    # Phase 4: Four-ring configuration
    bounds_4 = [(0.05, 1.5)] * 4 + [(0, 2*np.pi)] * 4
    try:
        de_res_4 = scipy.optimize.differential_evolution(
            _neg_score_four_ring_4433,
            bounds=bounds_4,
            seed=789,
            maxiter=60,
            tol=1e-8,
            polish=True,
            popsize=12
        )
        pts_4 = _four_ring(4, 4, 3, 3, *de_res_4.x)
        sc_4 = _score(_normalize(pts_4))
        _update_best(pts_4, sc_4)
    except Exception:
        pass

    if _time_left() < 12:
        return best_points if best_points is not None else _normalize(reg14)

    # Phase 5: Full 28D Nelder-Mead from best point
    if best_points is not None:
        try:
            res_full = scipy.optimize.minimize(
                _neg_score,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            pts_full = _normalize(res_full.x.reshape(-1, 2))
            sc_full = _score(pts_full)
            _update_best(pts_full, sc_full)
        except Exception:
            pass

    if _time_left() < 8:
        return best_points if best_points is not None else _normalize(reg14)

    # Phase 6: Smooth gradient-based polish with L-BFGS-B
    if best_points is not None and _time_left() > 5:
        try:
            init_flat = best_points.flatten()
            for beta in [50.0, 100.0, 200.0, 500.0]:
                if _time_left() < 3:
                    break
                res_polish = scipy.optimize.minimize(
                    lambda x, b=beta: _neg_smooth_score_and_grad(x, b),
                    init_flat,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
                )
                pts_polish = _normalize(res_polish.x.reshape(-1, 2))
                sc_polish = _score(pts_polish)
                _update_best(pts_polish, sc_polish)
                init_flat = res_polish.x
        except Exception:
            pass

    if _time_left() < 3:
        return best_points if best_points is not None else _normalize(reg14)

    # Phase 7: Final Nelder-Mead polish
    if best_points is not None:
        try:
            res_final = scipy.optimize.minimize(
                _neg_score,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
            )
            pts_final = _normalize(res_final.x.reshape(-1, 2))
            sc_final = _score(pts_final)
            _update_best(pts_final, sc_final)
        except Exception:
            pass

    if best_points is None:
        best_points = _normalize(reg14)
    else:
        best_points = _normalize(best_points)

    return best_points
