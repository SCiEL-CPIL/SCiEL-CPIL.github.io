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

def _generate_7_7(r, delta):
    """
    Generate 14 points: 7 on outer circle (radius 1) and 7 on inner circle (radius r).
    """
    n_outer = 7
    n_inner = 7
    
    angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    points_outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    
    angles_inner = np.linspace(delta, delta + 2 * np.pi, n_inner, endpoint=False)
    points_inner = np.column_stack([r * np.cos(angles_inner), r * np.sin(angles_inner)])
    
    return np.vstack([points_outer, points_inner])

def _generate_flexible_ring(r_inner, delta_inner, jitter_outer, jitter_inner):
    """
    Generate 14 points with flexible angular positions on two rings.
    """
    n_ring = 7
    base_angles = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
    
    outer_angles = base_angles + jitter_outer
    inner_angles = base_angles + delta_inner + jitter_inner
    
    points_outer = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    points_inner = np.column_stack([r_inner * np.cos(inner_angles), r_inner * np.sin(inner_angles)])
    
    return np.vstack([points_outer, points_inner])

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

    # Phase 1: Generate diverse initial candidates
    candidates = []
    
    # 1. Regular 14-gon
    candidates.append(_regular_polygon(14, 1.0))
    
    # 2. 7+7 structure with a few good (r, delta) guesses
    # Based on previous insights, r ~ 0.5-0.7, delta ~ pi/7 is good
    for r in [0.5, 0.6, 0.7, 0.8]:
        for delta in [0.0, np.pi/7, 2*np.pi/7]:
            candidates.append(_generate_7_7(r, delta))
            
    # 3. 13 on circle + 1 center
    center = np.array([[0.0, 0.0]])
    outer_13 = _regular_polygon(13, 1.0)
    candidates.append(np.vstack([center, outer_13]))
    
    # 4. Random perturbations of the best 7+7 found in a quick grid search
    # Quick grid search to find a good starting point for 7+7
    best_grid_r = 0.5
    best_grid_delta = 0.0
    best_grid_score = -1.0
    
    r_vals = np.linspace(0.3, 0.9, 10)
    delta_vals = np.linspace(0, 2 * np.pi / 7, 7, endpoint=False)
    
    for r in r_vals:
        for delta in delta_vals:
            pts = _generate_7_7(r, delta)
            sc = _score(_normalize(pts))
            if sc > best_grid_score:
                best_grid_score = sc
                best_grid_r = r
                best_grid_delta = delta
                # Add to candidates if significantly better
                if sc > best_score:
                    _update_best(pts, sc)
                    candidates.append(pts)
    
    if _time_left() < 20:
        return best_points if best_points is not None else _normalize(_regular_polygon(14, 1.0))

    # Phase 2: Refine top candidates using L-BFGS-B with soft surrogate
    # Sort candidates by score to prioritize the best ones
    scored_candidates = []
    for pts in candidates:
        sc = _score(_normalize(pts))
        scored_candidates.append((sc, pts))
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 3 unique candidates
    top_candidates = []
    for sc, pts in scored_candidates:
        if len(top_candidates) >= 3:
            break
        # Simple uniqueness check: distance to existing top candidates
        is_unique = True
        for existing in top_candidates:
            dist = np.linalg.norm(pts - existing)
            if dist < 0.1:
                is_unique = False
                break
        if is_unique:
            top_candidates.append(pts)
            
    if not top_candidates:
        top_candidates = [best_points if best_points is not None else _regular_polygon(14, 1.0)]

    for init_pts in top_candidates:
        if _time_left() < 10:
            break
            
        current_pts = init_pts.copy()
        
        # Progressive L-BFGS-B with increasing beta
        for beta in [20.0, 50.0, 100.0, 200.0]:
            if _time_left() < 5:
                break
            try:
                res = scipy.optimize.minimize(
                    lambda x, b=beta: _neg_soft_score(x, b),
                    current_pts.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                new_pts = _normalize(res.x.reshape(-1, 2))
                sc_new = _score(new_pts)
                if sc_new > best_score:
                    _update_best(new_pts, sc_new)
                    current_pts = new_pts
            except Exception:
                pass
        
        # Nelder-Mead polish on exact score
        if _time_left() > 5:
            try:
                res_nm = scipy.optimize.minimize(
                    _neg_score,
                    current_pts.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                new_pts = _normalize(res_nm.x.reshape(-1, 2))
                sc_new = _score(new_pts)
                if sc_new > best_score:
                    _update_best(new_pts, sc_new)
            except Exception:
                pass

    # Phase 3: If time remains, try a few random restarts with local optimization
    if _time_left() > 15 and best_points is not None:
        rng = np.random.RandomState(42)
        base_pts = best_points
        
        for _ in range(5):
            if _time_left() < 5:
                break
            # Perturb the best solution
            noise = rng.normal(0, 0.1, base_pts.shape)
            init_pts = base_pts + noise
            init_pts = _normalize(init_pts)
            
            try:
                # Quick L-BFGS-B
                res = scipy.optimize.minimize(
                    lambda x: _neg_soft_score(x, 100.0),
                    init_pts.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
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
