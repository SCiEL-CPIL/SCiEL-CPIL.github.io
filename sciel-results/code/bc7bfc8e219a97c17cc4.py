import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14

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

def _generate_two_ring_config(n1, n2, r1, r2, rot1, rot2):
    pts1 = _regular_polygon(n1, r1, rot1)
    pts2 = _regular_polygon(n2, r2, rot2)
    return np.vstack([pts1, pts2])

def _generate_three_ring_config(n1, n2, n3, r1, r2, r3, rot1, rot2, rot3):
    pts1 = _regular_polygon(n1, r1, rot1)
    pts2 = _regular_polygon(n2, r2, rot2)
    pts3 = _regular_polygon(n3, r3, rot3)
    return np.vstack([pts1, pts2, pts3])

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

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
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

    def _try_optimize_smooth(init_points, beta_start=30.0, beta_end=800.0, maxiter=3000):
        if time.time() > deadline:
            return
        init_norm = _normalize(init_points)
        x0 = init_norm.flatten()
        
        n_stages = 6
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
                    options={'maxiter': maxiter // n_stages, 'gtol': 1e-10, 'ftol': 1e-14}
                )
                x = res.x
                _update_best(x.reshape(-1, 2))
            except Exception:
                pass
        
        if time.time() < deadline - 3:
            try:
                def _obj_and_grad_high_beta(x_flat):
                    return _neg_smooth_score_and_grad(x_flat, 1500.0)
                
                res = scipy.optimize.minimize(
                    _obj_and_grad_high_beta,
                    x,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 800, 'gtol': 1e-12, 'ftol': 1e-16}
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
                        options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                    )
                    _update_best(res.x.reshape(-1, 2))
            except Exception:
                pass

    # 1. Explore two-ring configurations extensively
    splits = [(7, 7), (6, 8), (8, 6), (5, 9), (9, 5), (4, 10), (10, 4), (3, 11), (11, 3), (2, 12), (12, 2)]
    
    for n1, n2 in splits:
        if time.time() > deadline:
            break
        
        def _two_ring_score(params):
            r_ratio = params[0]
            rot1 = params[1]
            rot2 = params[2]
            
            r1 = 1.0
            r2 = r_ratio * r1
            
            if r2 <= 0.01:
                return -1e6
                
            pts = _generate_two_ring_config(n1, n2, r1, r2, rot1, rot2)
            sc = _score(_normalize(pts))
            return -sc

        for r_init in [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]:
            for rot_init in [0.0, np.pi/n1, np.pi/n2, 2*np.pi/n1, np.pi/(n1+n2), np.pi/n1 + np.pi/n2]:
                if time.time() > deadline:
                    continue
                x0 = np.array([r_init, 0.0, rot_init])
                try:
                    res = scipy.optimize.minimize(
                        _two_ring_score,
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
                    )
                    r_ratio, rot1, rot2 = res.x
                    pts = _generate_two_ring_config(n1, n2, 1.0, r_ratio, rot1, rot2)
                    sc = _update_best(pts)
                    
                    if sc > best_score * 0.85 and time.time() < deadline - 15:
                        _try_optimize_smooth(pts, maxiter=1500)
                        
                except Exception:
                    pass

    # 2. Explore three-ring configurations
    three_ring_splits = [(5, 4, 5), (4, 5, 5), (5, 5, 4), (6, 4, 4), (4, 6, 4), (4, 4, 6),
                         (7, 5, 2), (5, 7, 2), (2, 7, 5), (7, 2, 5), (2, 5, 7), (5, 2, 7),
                         (6, 6, 2), (6, 2, 6), (2, 6, 6)]
    
    for n1, n2, n3 in three_ring_splits:
        if time.time() > deadline:
            break
        
        def _three_ring_score(params):
            r1, r2, r3 = params[0], params[1], params[2]
            rot1, rot2, rot3 = params[3], params[4], params[5]
            
            if r1 <= 0.01 or r2 <= 0.01 or r3 <= 0.01:
                return -1e6
                
            pts = _generate_three_ring_config(n1, n2, n3, r1, r2, r3, rot1, rot2, rot3)
            sc = _score(_normalize(pts))
            return -sc

        # Use a few reasonable starting points for each split
        r_starts = [
            [1.0, 0.5, 0.2],
            [1.0, 0.6, 0.3],
            [1.0, 0.7, 0.4],
            [1.0, 0.4, 0.15]
        ]
        
        for r_start in r_starts:
            if time.time() > deadline:
                break
            
            rot_starts = [
                [0.0, 0.0, 0.0],
                [0.0, np.pi/n2, 0.0],
                [np.pi/n1, 0.0, np.pi/n3],
                [0.0, 2*np.pi/n2, 2*np.pi/n3]
            ]
            
            for rot_start in rot_starts:
                if time.time() > deadline:
                    break
                
                x0 = np.array(r_start + rot_start)
                try:
                    res = scipy.optimize.minimize(
                        _three_ring_score,
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12}
                    )
                    r1, r2, r3, rot1, rot2, rot3 = res.x
                    pts = _generate_three_ring_config(n1, n2, n3, r1, r2, r3, rot1, rot2, rot3)
                    sc = _update_best(pts)
                    
                    if sc > best_score * 0.85 and time.time() < deadline - 15:
                        _try_optimize_smooth(pts, maxiter=1500)
                        
                except Exception:
                    pass

    # 3. If we still have time, do a few random restarts from scratch with smooth optimization
    if time.time() < deadline - 20:
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            
            # Generate random convex-ish points
            angles = np.sort(rng.uniform(0, 2 * np.pi, N))
            radii = rng.uniform(0.5, 1.0, N)
            pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            
            sc = _update_best(pts)
            if sc > best_score * 0.5 and time.time() < deadline - 10:
                _try_optimize_smooth(pts, maxiter=2000)

    if best_points is None:
        best_points = _normalize(_regular_polygon(14, 1.0, 0.0))
    else:
        best_points = _normalize(best_points)
        
    return best_points
