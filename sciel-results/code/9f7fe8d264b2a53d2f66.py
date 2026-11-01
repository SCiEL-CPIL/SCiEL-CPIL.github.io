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
    if max_a < 1e-12:
        return 0.0
    shifted = areas - max_a
    log_sum = np.log(np.sum(np.exp(-beta * shifted))) / beta
    soft_min = max_a - log_sum
    return soft_min

def _hull_area(points):
    """Compute convex hull area."""
    try:
        # Check for collinearity or degenerate hull
        if len(np.unique(points, axis=0)) < 3:
            return 0.0
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

def _neg_score(points_flat):
    """Negative score for minimization."""
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
    """Regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _two_ring_points(r, delta, jitter_outer=None, jitter_inner=None):
    """Generate 14 points: 7 outer, 7 inner with optional jitter."""
    n_ring = 7
    base_angles = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
    
    if jitter_outer is not None:
        outer_angles = base_angles + jitter_outer
    else:
        outer_angles = base_angles
    
    if jitter_inner is not None:
        inner_angles = base_angles + delta + jitter_inner
    else:
        inner_angles = base_angles + delta
    
    points_outer = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    points_inner = np.column_stack([r * np.cos(inner_angles), r * np.sin(inner_angles)])
    
    return np.vstack([points_outer, points_inner])

def _neg_score_two_ring(params):
    """Negative score for two-ring parameterization. params: [r, delta, 7 outer jitters, 7 inner jitters]"""
    try:
        r = params[0]
        delta = params[1]
        jitter_outer = params[2:9]
        jitter_inner = params[9:16]
        r = np.clip(r, 0.1, 0.95)
        pts = _two_ring_points(r, delta, jitter_outer, jitter_inner)
        ha = _hull_area(pts)
        if ha < 1e-12:
            return 0.0
        mta = _min_triangle_area(pts)
        return -(mta / ha)
    except Exception:
        return 0.0

def _neg_soft_score_two_ring(params, beta=100.0):
    """Negative soft score for two-ring parameterization."""
    try:
        r = params[0]
        delta = params[1]
        jitter_outer = params[2:9]
        jitter_inner = params[9:16]
        r = np.clip(r, 0.1, 0.95)
        pts = _two_ring_points(r, delta, jitter_outer, jitter_inner)
        ha = _hull_area(pts)
        if ha < 1e-12:
            return 0.0
        smta = _soft_min_triangle_area(pts, beta)
        return -(smta / ha)
    except Exception:
        return 0.0

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

    def _elapsed():
        return time.time() - start_time

    def _time_left():
        return deadline - time.time()

    # Phase 1: Fine grid search for (r, delta)
    best_grid_r = 0.5
    best_grid_delta = 0.0
    best_grid_score = -1.0
    
    for r in np.linspace(0.3, 0.9, 15):
        for delta in np.linspace(0, 2 * np.pi / 7, 10, endpoint=False):
            pts = _two_ring_points(r, delta)
            sc = _score(_normalize(pts))
            if sc > best_grid_score:
                best_grid_score = sc
                best_grid_r = r
                best_grid_delta = delta
            if sc > best_score:
                _update_best(pts, sc)
    
    if _time_left() < 20:
        return best_points if best_points is not None else _normalize(_regular_polygon(14, 1.0))

    # Phase 2: Optimize in two-ring parameter space (16D)
    n_params = 16
    init_params = np.zeros(n_params)
    init_params[0] = best_grid_r
    init_params[1] = best_grid_delta
    
    for beta in [20.0, 50.0, 100.0, 200.0, 500.0]:
        if _time_left() < 5:
            break
        try:
            res = scipy.optimize.minimize(
                lambda x, b=beta: _neg_soft_score_two_ring(x, b),
                init_params,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            r = np.clip(res.x[0], 0.1, 0.95)
            delta = res.x[1]
            jitter_outer = res.x[2:9]
            jitter_inner = res.x[9:16]
            pts = _two_ring_points(r, delta, jitter_outer, jitter_inner)
            pts = _normalize(pts)
            sc = _score(pts)
            if sc > best_score:
                _update_best(pts, sc)
                init_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Full-space refinement
    if best_points is not None and _time_left() > 10:
        current_pts = best_points.copy()
        
        for beta in [100.0, 200.0, 500.0]:
            if _time_left() < 5:
                break
            try:
                res = scipy.optimize.minimize(
                    lambda x, b=beta: _neg_soft_score(x, b),
                    current_pts.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                new_pts = _normalize(res.x.reshape(-1, 2))
                sc_new = _score(new_pts)
                if sc_new > best_score:
                    _update_best(new_pts, sc_new)
                    current_pts = new_pts
            except Exception:
                pass
        
        # Nelder-Mead polish
        if _time_left() > 5:
            try:
                res_nm = scipy.optimize.minimize(
                    _neg_score,
                    current_pts.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10}
                )
                new_pts = _normalize(res_nm.x.reshape(-1, 2))
                sc_new = _score(new_pts)
                if sc_new > best_score:
                    _update_best(new_pts, sc_new)
            except Exception:
                pass

    # Phase 4: Random restarts with smaller noise
    if _time_left() > 15 and best_points is not None:
        rng = np.random.RandomState(42)
        base_pts = best_points
        
        for _ in range(8):
            if _time_left() < 5:
                break
            noise = rng.normal(0, 0.05, base_pts.shape)
            init_pts = _normalize(base_pts + noise)
            
            try:
                res = scipy.optimize.minimize(
                    lambda x: _neg_soft_score(x, 200.0),
                    init_pts.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                new_pts = _normalize(res.x.reshape(-1, 2))
                sc_new = _score(new_pts)
                if sc_new > best_score:
                    _update_best(new_pts, sc_new)
            except Exception:
                pass

    if best_points is None:
        best_points = _normalize(_regular_polygon(14, 1.0))
    else:
        best_points = _normalize(best_points)
        
    return best_points
