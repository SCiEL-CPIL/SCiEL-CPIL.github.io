import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14
DEADLINE = 140.0  # safety margin within 150s kill limit

def _get_triangle_indices(n):
    indices = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
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

def _soft_min_triangle_area(points, beta):
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    max_a = np.max(areas)
    if max_a < 1e-15:
        return 0.0
    # Use log-sum-exp trick for numerical stability
    shifted = -beta * (areas - max_a)
    # Clip to avoid overflow
    shifted = np.clip(shifted, -500, 500)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_a - log_sum / beta
    return soft_min

def _neg_soft_score(points_flat, beta=100.0):
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

def _generate_rings(sizes, radii, rotations):
    """Generate points as union of concentric regular polygons."""
    pts_list = []
    for n, r, rot in zip(sizes, radii, rotations):
        if n > 0:
            pts_list.append(_regular_polygon(n, r, rot))
    return np.vstack(pts_list)

def _neg_ring_score(params, sizes):
    """Negative score for ring parameterization.
    params: [r1, r2, ..., rK, rot1, rot2, ..., rotK]
    """
    try:
        K = len(sizes)
        radii = params[:K]
        rotations = params[K:2*K]
        
        # Ensure radii are positive
        radii = np.maximum(radii, 1e-6)
        
        pts = _generate_rings(sizes, radii, rotations)
        pts_norm = _normalize(pts)
        return -_score(pts_norm)
    except Exception:
        return 0.0

def _neg_soft_ring_score(params, sizes, beta=100.0):
    """Negative soft-score for ring parameterization (for gradient-based optimization)."""
    try:
        K = len(sizes)
        radii = params[:K]
        rotations = params[K:2*K]
        
        radii = np.maximum(radii, 1e-6)
        
        pts = _generate_rings(sizes, radii, rotations)
        ha = _hull_area(pts)
        if ha < 1e-12:
            return 0.0
        smta = _soft_min_triangle_area(pts, beta)
        return -(smta / ha)
    except Exception:
        return 0.0

def _time_left(deadline):
    return deadline - time.time()

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_score = 0.0
    best_points = None
    
    def _update_best(pts, sc):
        nonlocal best_score, best_points
        if sc > best_score:
            best_score = sc
            best_points = _normalize(pts).copy()
    
    # Try multiple ring configurations
    ring_configs = [
        ([7, 7], "two-ring 7/7"),
        ([7, 5, 2], "three-ring 7/5/2"),
        ([6, 6, 2], "three-ring 6/6/2"),
        ([5, 5, 4], "three-ring 5/5/4"),
        ([4, 4, 3, 3], "four-ring 4/4/3/3"),
        ([5, 4, 3, 2], "four-ring 5/4/3/2"),
    ]
    
    # Phase 1: Coarse grid search for each ring config
    for sizes, name in ring_configs:
        if _time_left(deadline) < 30:
            break
        
        K = len(sizes)
        
        # Grid search over radii (log-spaced) and rotations
        # For each ring, search radius in [0.2, 1.5] and rotation in [0, 2*pi/n]
        r_grid = np.logspace(np.log10(0.2), np.log10(1.5), 8)
        
        # For simplicity, do a small grid for each ring
        # Start with all rotations at 0, then vary
        best_ring_score = 0.0
        best_ring_params = None
        
        # Create a grid for the first ring's radius and rotation,
        # then for subsequent rings
        # For efficiency, do a reduced grid
        if K == 2:
            # Two rings: 8x8x8x8 = 4096 evaluations might be too many
            # Do 5x5x5x5 = 625
            r1_vals = np.logspace(np.log10(0.3), np.log10(1.2), 5)
            r2_vals = np.logspace(np.log10(0.3), np.log10(1.0), 5)
            rot1_vals = np.linspace(0, 2 * np.pi / sizes[0], 5, endpoint=False)
            rot2_vals = np.linspace(0, 2 * np.pi / sizes[1], 5, endpoint=False)
            
            for r1 in r1_vals:
                for r2 in r2_vals:
                    for rot1 in rot1_vals:
                        for rot2 in rot2_vals:
                            if _time_left(deadline) < 25:
                                break
                            params = [r1, r2, rot1, rot2]
                            sc = -_neg_ring_score(params, sizes)
                            if sc > best_ring_score:
                                best_ring_score = sc
                                best_ring_params = params
        elif K == 3:
            # Three rings: do a smaller grid
            r1_vals = np.logspace(np.log10(0.4), np.log10(1.3), 4)
            r2_vals = np.logspace(np.log10(0.3), np.log10(0.9), 4)
            r3_vals = np.logspace(np.log10(0.2), np.log10(0.7), 4)
            rot1_vals = np.linspace(0, 2 * np.pi / sizes[0], 3, endpoint=False)
            rot2_vals = np.linspace(0, 2 * np.pi / sizes[1], 3, endpoint=False)
            rot3_vals = np.linspace(0, 2 * np.pi / sizes[2], 3, endpoint=False)
            
            for r1 in r1_vals:
                for r2 in r2_vals:
                    for r3 in r3_vals:
                        for rot1 in rot1_vals:
                            for rot2 in rot2_vals:
                                for rot3 in rot3_vals:
                                    if _time_left(deadline) < 20:
                                        break
                                    params = [r1, r2, r3, rot1, rot2, rot3]
                                    sc = -_neg_ring_score(params, sizes)
                                    if sc > best_ring_score:
                                        best_ring_score = sc
                                        best_ring_params = params
        elif K == 4:
            # Four rings: even smaller grid
            r_vals = [
                np.logspace(np.log10(0.5), np.log10(1.3), 3),
                np.logspace(np.log10(0.4), np.log10(1.0), 3),
                np.logspace(np.log10(0.3), np.log10(0.8), 3),
                np.logspace(np.log10(0.2), np.log10(0.6), 3),
            ]
            rot_vals = [np.linspace(0, 2 * np.pi / n, 2, endpoint=False) for n in sizes]
            
            for r1 in r_vals[0]:
                for r2 in r_vals[1]:
                    for r3 in r_vals[2]:
                        for r4 in r_vals[3]:
                            for rot1 in rot_vals[0]:
                                for rot2 in rot_vals[1]:
                                    for rot3 in rot_vals[2]:
                                        for rot4 in rot_vals[3]:
                                            if _time_left(deadline) < 15:
                                                break
                                            params = [r1, r2, r3, r4, rot1, rot2, rot3, rot4]
                                            sc = -_neg_ring_score(params, sizes)
                                            if sc > best_ring_score:
                                                best_ring_score = sc
                                                best_ring_params = params
        
        if best_ring_params is not None:
            radii = best_ring_params[:K]
            rotations = best_ring_params[K:2*K]
            pts = _generate_rings(sizes, radii, rotations)
            _update_best(pts, best_ring_score)
    
    if best_points is None:
        best_points = _normalize(_regular_polygon(14, 1.0, 0.0))
        best_score = _score(best_points)
    
    if _time_left(deadline) < 40:
        return best_points
    
    # Phase 2: Differential Evolution for each ring config
    for sizes, name in ring_configs:
        if _time_left(deadline) < 30:
            break
        
        K = len(sizes)
        bounds = [(0.1, 1.5)] * K + [(0, 2 * np.pi)] * K
        
        try:
            de_res = scipy.optimize.differential_evolution(
                lambda x: _neg_ring_score(x, sizes),
                bounds=bounds,
                seed=42 + K,
                maxiter=80,
                tol=1e-6,
                polish=False,
                popsize=15
            )
            sc_de = -de_res.fun
            radii = de_res.x[:K]
            rotations = de_res.x[K:2*K]
            pts = _generate_rings(sizes, radii, rotations)
            _update_best(pts, sc_de)
        except Exception:
            pass
    
    if _time_left(deadline) < 25:
        return best_points
    
    # Phase 3: L-BFGS-B polish using soft surrogate for each ring config
    for sizes, name in ring_configs:
        if _time_left(deadline) < 20:
            break
        
        K = len(sizes)
        
        # Try multiple starting points
        start_points = []
        # Start from regular polygon radii
        r_init = np.linspace(1.0, 0.3, K)
        rot_init = np.zeros(K)
        start_points.append(np.concatenate([r_init, rot_init]))
        
        # Start from a spread
        r_init2 = np.array([1.2, 0.8, 0.5, 0.3][:K])
        rot_init2 = np.array([0, np.pi / sizes[1] if K > 1 else 0, 
                              np.pi / sizes[2] if K > 2 else 0,
                              np.pi / sizes[3] if K > 3 else 0])
        start_points.append(np.concatenate([r_init2, rot_init2]))
        
        for beta in [50.0, 100.0, 200.0]:
            if _time_left(deadline) < 15:
                break
            
            for sp in start_points:
                if _time_left(deadline) < 10:
                    break
                try:
                    res_lbfgs = scipy.optimize.minimize(
                        lambda x, b=beta: _neg_soft_ring_score(x, sizes, b),
                        sp,
                        method='L-BFGS-B',
                        options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    sc_lbfgs = -res_lbfgs.fun
                    radii = res_lbfgs.x[:K]
                    rotations = res_lbfgs.x[K:2*K]
                    pts = _generate_rings(sizes, radii, rotations)
                    _update_best(pts, sc_lbfgs)
                    # Use result as next start point
                    start_points.append(res_lbfgs.x.copy())
                except Exception:
                    pass
    
    if _time_left(deadline) < 15:
        return best_points
    
    # Phase 4: Full 28D Nelder-Mead from best point
    if best_points is not None:
        try:
            res_full = scipy.optimize.minimize(
                _neg_score,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            pts_full = _normalize(res_full.x.reshape(-1, 2))
            sc_full = _score(pts_full)
            _update_best(pts_full, sc_full)
        except Exception:
            pass
    
    if _time_left(deadline) < 8:
        return best_points
    
    # Phase 5: Final L-BFGS-B polish on full 28D
    if best_points is not None and _time_left(deadline) > 5:
        for beta in [100.0, 200.0, 500.0]:
            if _time_left(deadline) < 4:
                break
            try:
                res_polish = scipy.optimize.minimize(
                    lambda x, b=beta: _neg_soft_score(x, b),
                    best_points.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-13, 'gtol': 1e-12}
                )
                pts_polish = _normalize(res_polish.x.reshape(-1, 2))
                sc_polish = _score(pts_polish)
                _update_best(pts_polish, sc_polish)
            except Exception:
                pass
    
    if best_points is None:
        best_points = _normalize(_regular_polygon(14, 1.0, 0.0))
    else:
        best_points = _normalize(best_points)
    
    return best_points
