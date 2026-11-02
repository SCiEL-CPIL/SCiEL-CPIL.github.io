import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14

# Pre-compute triangle indices for speed
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
_N_TRI = len(_TRI_INDICES)

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

def _two_heptagons(r1, r2, rot1, rot2):
    """Generate 14 points as two regular heptagons with given radii and rotations."""
    pts1 = _regular_polygon(7, r1, rot1)
    pts2 = _regular_polygon(7, r2, rot2)
    return np.vstack([pts1, pts2])

def _three_ring(r1, r2, r3, rot1, rot2, rot3, n1=5, n2=5, n3=4):
    """Generate 14 points on three concentric regular polygons."""
    pts1 = _regular_polygon(n1, r1, rot1)
    pts2 = _regular_polygon(n2, r2, rot2)
    pts3 = _regular_polygon(n3, r3, rot3)
    return np.vstack([pts1, pts2, pts3])

def _neg_two_heptagons_score(params):
    """Negative score for two-heptagon configuration."""
    r1, r2, rot1, rot2 = params
    if r1 < 0.01 or r2 < 0.01:
        return 1e6
    pts = _two_heptagons(r1, r2, rot1, rot2)
    sc = _score(_normalize(pts))
    return -sc

def _neg_three_ring_score(params):
    """Negative score for three-ring configuration."""
    r1, r2, r3, rot1, rot2, rot3 = params
    if r1 < 0.01 or r2 < 0.01 or r3 < 0.01:
        return 1e6
    pts = _three_ring(r1, r2, r3, rot1, rot2, rot3)
    sc = _score(_normalize(pts))
    return -sc

def _neg_smooth_score_and_grad(points_flat, beta=100.0):
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

    shifted = areas - max_area
    exp_vals = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_vals)
    if sum_exp == 0:
        return 0.0, np.zeros_like(points_flat)
    weights = exp_vals / sum_exp
    
    log_sum = -beta * max_area + np.log(sum_exp)
    smooth_min = -log_sum / beta
    
    ha = _hull_area(points)
    if ha < 1e-12:
        return 0.0, np.zeros_like(points_flat)
    
    score = smooth_min / ha
    
    signs = np.sign(cross)
    
    d_cross_di_x = -(v1[:, 1] + v2[:, 1])
    d_cross_di_y = (v1[:, 0] + v2[:, 0])
    d_cross_dj_x = v2[:, 1]
    d_cross_dj_y = -v2[:, 0]
    d_cross_dk_x = -v1[:, 1]
    d_cross_dk_y = v1[:, 0]
    
    w = weights * signs * 0.5
    
    grad_points = np.zeros_like(points)
    
    dA_di = w[:, None] * np.column_stack([d_cross_di_x, d_cross_di_y])
    dA_dj = w[:, None] * np.column_stack([d_cross_dj_x, d_cross_dj_y])
    dA_dk = w[:, None] * np.column_stack([d_cross_dk_x, d_cross_dk_y])
    
    np.add.at(grad_points, _I, dA_di)
    np.add.at(grad_points, _J, dA_dj)
    np.add.at(grad_points, _K, dA_dk)
    
    grad_score = grad_points / ha
    
    return -score, -grad_score.flatten()

def _ring_config(r1, r2, rot1, rot2, edge_params):
    """
    Generate 14 points: 7 on outer heptagon (radius r1, rotation rot1)
    and 7 on edges of inner heptagon (radius r2, rotation rot2).
    edge_params: array of 7 values in [0,1], each specifying position on an edge.
    """
    # Outer heptagon vertices
    angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False) + rot1
    outer_pts = np.column_stack([r1 * np.cos(angles_outer), r1 * np.sin(angles_outer)])
    
    # Inner heptagon vertices (for edge definitions)
    angles_inner = np.linspace(0, 2 * np.pi, 7, endpoint=False) + rot2
    inner_verts = np.column_stack([r2 * np.cos(angles_inner), r2 * np.sin(angles_inner)])
    
    # Points on edges of inner heptagon
    edge_pts = np.zeros((7, 2))
    for i in range(7):
        t = edge_params[i]
        v1 = inner_verts[i]
        v2 = inner_verts[(i + 1) % 7]
        edge_pts[i] = (1 - t) * v1 + t * v2
    
    return np.vstack([outer_pts, edge_pts])

def _neg_ring_score(params):
    """Negative score for ring configuration."""
    r1, r2, rot1, rot2 = params[:4]
    edge_params = params[4:]
    if r1 < 0.01 or r2 < 0.01:
        return 1e6
    pts = _ring_config(r1, r2, rot1, rot2, edge_params)
    sc = _score(_normalize(pts))
    return -sc

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    best_score = 0.0
    best_points = None
    rng = np.random.RandomState(42)

    def _update_best(points):
        nonlocal best_score, best_points
        pts = _normalize(points)
        sc = _score(pts)
        if sc > best_score:
            best_score = sc
            best_points = pts.copy()
        return sc

    def _try_optimize_smooth(init_points, beta_start=50.0, beta_end=500.0, maxiter=2000):
        if time.time() > deadline:
            return
        init_norm = _normalize(init_points)
        x0 = init_norm.flatten()
        
        n_stages = 5
        betas = np.linspace(beta_start, beta_end, n_stages)
        
        x = x0.copy()
        for beta in betas:
            if time.time() > deadline - 5:
                break
            
            def _obj_and_grad(x_flat, beta_val=beta):
                return _neg_smooth_score_and_grad(x_flat, beta_val)
            
            try:
                res = scipy.optimize.minimize(
                    _obj_and_grad,
                    x,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': maxiter // n_stages, 'gtol': 1e-8, 'ftol': 1e-12}
                )
                x = res.x
                _update_best(x.reshape(-1, 2))
            except Exception:
                pass
        
        if time.time() < deadline - 3:
            try:
                def _obj_and_grad_high_beta(x_flat):
                    return _neg_smooth_score_and_grad(x_flat, 1000.0)
                
                res = scipy.optimize.minimize(
                    _obj_and_grad_high_beta,
                    x,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-14}
                )
                x = res.x
                _update_best(x.reshape(-1, 2))
                
                if time.time() < deadline - 1:
                    def _neg_exact_score(x_flat):
                        return -_score(x_flat.reshape(-1, 2))
                    
                    res = scipy.optimize.minimize(
                        _neg_exact_score,
                        x,
                        method='Nelder-Mead',
                        options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                    )
                    _update_best(res.x.reshape(-1, 2))
            except Exception:
                pass

    def _coordinate_descent_polish(pts, max_rounds=5):
        """Polish using coordinate descent on the exact score."""
        if time.time() > deadline - 5:
            return pts
        for iteration in range(max_rounds):
            if time.time() > deadline - 3:
                break
            for i in range(N):
                if time.time() > deadline - 3:
                    break
                def _neg_score_one_point(params):
                    new_pts = pts.copy()
                    new_pts[i, 0] = params[0]
                    new_pts[i, 1] = params[1]
                    return -_score(new_pts)
                
                x0 = np.array([pts[i, 0], pts[i, 1]])
                try:
                    res = scipy.optimize.minimize(
                        _neg_score_one_point,
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 80, 'xatol': 1e-11, 'fatol': 1e-13}
                    )
                    pts[i, 0] = res.x[0]
                    pts[i, 1] = res.x[1]
                except Exception:
                    pass
        return pts

    # === PHASE 1: Two-heptagon differential evolution ===
    bounds_h = [(0.1, 1.5), (0.1, 1.5), (-np.pi/7, np.pi/7), (-np.pi/7, np.pi/7)]
    
    try:
        de_res = scipy.optimize.differential_evolution(
            _neg_two_heptagons_score,
            bounds_h,
            maxiter=300,
            tol=1e-10,
            popsize=20,
            seed=42,
            polish=True,
            init='sobol'
        )
        best_params_de = de_res.x
        best_pts_de = _two_heptagons(*best_params_de)
        sc_de = _update_best(best_pts_de)
    except Exception:
        best_params_de = None

    # === PHASE 2: Three-ring differential evolution ===
    bounds_3r = [(0.1, 1.5), (0.1, 1.2), (0.05, 0.9),
                 (-np.pi/5, np.pi/5), (-np.pi/5, np.pi/5), (-np.pi/4, np.pi/4)]
    
    if time.time() < deadline - 30:
        try:
            de_res_3r = scipy.optimize.differential_evolution(
                _neg_three_ring_score,
                bounds_3r,
                maxiter=200,
                tol=1e-8,
                popsize=15,
                seed=42,
                polish=True,
                init='sobol'
            )
            best_params_3r = de_res_3r.x
            pts_3r = _three_ring(*best_params_3r)
            sc_3r = _update_best(pts_3r)
        except Exception:
            pass

    # === PHASE 3: Ring configuration (outer heptagon + inner edge points) ===
    bounds_ring = [(0.1, 1.5), (0.05, 1.2), (-np.pi/7, np.pi/7), (-np.pi/7, np.pi/7)] + \
                  [(0.0, 1.0)] * 7
    
    if time.time() < deadline - 25:
        try:
            de_res_ring = scipy.optimize.differential_evolution(
                _neg_ring_score,
                bounds_ring,
                maxiter=150,
                tol=1e-8,
                popsize=15,
                seed=42,
                polish=True,
                init='sobol'
            )
            best_params_ring = de_res_ring.x
            r1, r2, rot1, rot2 = best_params_ring[:4]
            edge_params = best_params_ring[4:]
            pts_ring = _ring_config(r1, r2, rot1, rot2, edge_params)
            sc_ring = _update_best(pts_ring)
        except Exception:
            pass

    # === PHASE 4: Local refinement from multiple starting points ===
    if time.time() < deadline - 20:
        symmetric_starts_2h = [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, np.pi/7],
            [1.0, 0.5, 0.0, 0.0],
            [1.0, 0.5, 0.0, np.pi/7],
            [1.0, 0.3, 0.0, np.pi/14],
            [1.0, 0.7, 0.0, np.pi/7],
            [0.8, 0.8, 0.0, np.pi/7],
            [1.0, 0.4, np.pi/14, 0.0],
        ]
        
        for x0 in symmetric_starts_2h:
            if time.time() > deadline - 15:
                break
            try:
                res = scipy.optimize.minimize(
                    _neg_two_heptagons_score,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-14}
                )
                pts = _two_heptagons(*res.x)
                sc = _update_best(pts)
                
                if sc > best_score * 0.95 and time.time() < deadline - 15:
                    _try_optimize_smooth(pts, maxiter=1500)
            except Exception:
                pass

        # Three-ring starts
        symmetric_starts_3r = [
            [1.0, 0.6, 0.3, 0.0, 0.0, 0.0],
            [1.0, 0.6, 0.3, 0.0, np.pi/5, 0.0],
            [1.0, 0.7, 0.4, 0.0, 0.0, np.pi/4],
            [1.0, 0.5, 0.25, np.pi/5, 0.0, 0.0],
        ]
        
        for x0 in symmetric_starts_3r:
            if time.time() > deadline - 15:
                break
            try:
                res = scipy.optimize.minimize(
                    _neg_three_ring_score,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-14}
                )
                pts = _three_ring(*res.x)
                sc = _update_best(pts)
                
                if sc > best_score * 0.95 and time.time() < deadline - 15:
                    _try_optimize_smooth(pts, maxiter=1500)
            except Exception:
                pass

    # === PHASE 5: Full 2D optimization from best results ===
    if best_points is not None and time.time() < deadline - 15:
        _try_optimize_smooth(best_points, beta_start=100.0, beta_end=1000.0, maxiter=2000)

    # === PHASE 6: Random restarts ===
    for i in range(6):
        if time.time() > deadline - 10:
            break
        r1 = rng.uniform(0.5, 1.5)
        r2 = rng.uniform(0.5, 1.5)
        rot1 = rng.uniform(-np.pi/7, np.pi/7)
        rot2 = rng.uniform(-np.pi/7, np.pi/7)
        init = _two_heptagons(r1, r2, rot1, rot2)
        
        noise = rng.normal(0, 0.05, (N, 2))
        init = init + noise
        
        if time.time() < deadline - 12:
            _try_optimize_smooth(init, maxiter=1000)

    # === PHASE 7: Coordinate descent polish ===
    if best_points is not None and time.time() < deadline - 8:
        pts = best_points.copy()
        pts = _coordinate_descent_polish(pts, max_rounds=4)
        _update_best(pts)

    # === PHASE 8: High-precision final polish ===
    if best_points is not None and time.time() < deadline - 4:
        x0 = best_points.flatten()
        try:
            def _obj_and_grad_final(x_flat):
                return _neg_smooth_score_and_grad(x_flat, 2000.0)
            
            res = scipy.optimize.minimize(
                _obj_and_grad_final,
                x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 300, 'gtol': 1e-12, 'ftol': 1e-15}
            )
            _update_best(res.x.reshape(-1, 2))
        except Exception:
            pass

    # === PHASE 9: Final coordinate descent ===
    if best_points is not None and time.time() < deadline - 2:
        pts = best_points.copy()
        pts = _coordinate_descent_polish(pts, max_rounds=3)
        _update_best(pts)

    if best_points is None:
        best_points = _normalize(_regular_polygon(N, 1.0))
    else:
        best_points = _normalize(best_points)
        
    final_score = _score(best_points)
    if final_score < 1e-10:
        best_points = _normalize(_regular_polygon(N, 1.0))
        
    return best_points
