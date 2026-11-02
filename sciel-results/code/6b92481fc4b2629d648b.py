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

def _generate_four_ring_config(n1, n2, n3, n4, r1, r2, r3, r4, rot1, rot2, rot3, rot4):
    pts1 = _regular_polygon(n1, r1, rot1)
    pts2 = _regular_polygon(n2, r2, rot2)
    pts3 = _regular_polygon(n3, r3, rot3)
    pts4 = _regular_polygon(n4, r4, rot4)
    return np.vstack([pts1, pts2, pts3, pts4])

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

def _smooth_min_value(points, beta=100.0):
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    max_area = np.max(areas)
    if max_area < 1e-15:
        return 0.0
    shifted = areas - max_area
    exp_vals = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_vals)
    if sum_exp == 0:
        return 0.0
    log_sum = -beta * max_area + np.log(sum_exp)
    return -log_sum / beta

def construct():
    start_time = time.time()
    deadline = start_time + 138.0  # Safety margin before 150s kill
    
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

    def _try_optimize_smooth(init_points, beta_start=20.0, beta_end=500.0, maxiter=2000):
        if time.time() > deadline - 5:
            return
        init_norm = _normalize(init_points)
        x0 = init_norm.flatten()
        
        # Fewer stages, more iterations per stage for better convergence
        n_stages = 4
        betas = np.geomspace(beta_start, beta_end, n_stages)
        
        x = x0.copy()
        for beta in betas:
            if time.time() > deadline - 10:
                break
            
            def _obj_and_grad(x_flat, beta_val=beta):
                return _neg_smooth_score_and_grad(x_flat, beta_val)
            
            try:
                res = scipy.optimize.minimize(
                    _obj_and_grad,
                    x,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': maxiter // n_stages, 'gtol': 1e-12, 'ftol': 1e-16}
                )
                x = res.x
                _update_best(x.reshape(-1, 2))
            except Exception:
                pass
        
        # Final high-precision polish with very high beta smooth surrogate
        if time.time() < deadline - 8:
            try:
                def _obj_and_grad_high_beta(x_flat):
                    return _neg_smooth_score_and_grad(x_flat, 2000.0)
                
                res = scipy.optimize.minimize(
                    _obj_and_grad_high_beta,
                    x,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'gtol': 1e-14, 'ftol': 1e-18}
                )
                x = res.x
                _update_best(x.reshape(-1, 2))
            except Exception:
                pass

    # 1. Explore two-ring configurations with reduced but strategic starts
    splits = [(7, 7), (6, 8), (8, 6), (5, 9), (9, 5)]
    
    for n1, n2 in splits:
        if time.time() > deadline - 30:
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

        # Fewer, more diverse starting points
        r_inits = [0.3, 0.5, 0.7, 0.9]
        rot_inits = [0.0, np.pi/n1, np.pi/n2, (np.pi/n1 + np.pi/n2)/2]
        
        for r_init in r_inits:
            for rot_init in rot_inits:
                if time.time() > deadline - 30:
                    continue
                x0 = np.array([r_init, 0.0, rot_init])
                try:
                    res = scipy.optimize.minimize(
                        _two_ring_score,
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12}
                    )
                    r_ratio, rot1, rot2 = res.x
                    if r_ratio <= 0.01 or r_ratio > 2.0:
                        continue
                    pts = _generate_two_ring_config(n1, n2, 1.0, r_ratio, rot1, rot2)
                    sc = _update_best(pts)
                    
                    if sc > best_score * 0.85 and time.time() < deadline - 25:
                        _try_optimize_smooth(pts, maxiter=1200)
                        
                except Exception:
                    pass

    # 2. Explore three-ring configurations
    three_ring_splits = [(5, 4, 5), (4, 5, 5), (5, 5, 4), (6, 4, 4), (4, 6, 4), (4, 4, 6)]
    
    for n1, n2, n3 in three_ring_splits:
        if time.time() > deadline - 30:
            break
        
        def _three_ring_score(params):
            r1, r2, r3 = params[0], params[1], params[2]
            rot1, rot2, rot3 = params[3], params[4], params[5]
            
            if r1 <= 0.01 or r2 <= 0.01 or r3 <= 0.01:
                return -1e6
                
            pts = _generate_three_ring_config(n1, n2, n3, r1, r2, r3, rot1, rot2, rot3)
            sc = _score(_normalize(pts))
            return -sc

        r_starts = [
            [1.0, 0.5, 0.2],
            [1.0, 0.6, 0.35],
            [1.0, 0.4, 0.15]
        ]
        
        for r_start in r_starts:
            if time.time() > deadline - 30:
                break
            
            rot_starts = [
                [0.0, 0.0, 0.0],
                [0.0, np.pi/n2, 0.0],
                [np.pi/n1, 0.0, np.pi/n3]
            ]
            
            for rot_start in rot_starts:
                if time.time() > deadline - 30:
                    break
                
                x0 = np.array(r_start + rot_start)
                try:
                    res = scipy.optimize.minimize(
                        _three_ring_score,
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 250, 'xatol': 1e-10, 'fatol': 1e-12}
                    )
                    r1, r2, r3, rot1, rot2, rot3 = res.x
                    if r1 <= 0.01 or r2 <= 0.01 or r3 <= 0.01:
                        continue
                    pts = _generate_three_ring_config(n1, n2, n3, r1, r2, r3, rot1, rot2, rot3)
                    sc = _update_best(pts)
                    
                    if sc > best_score * 0.85 and time.time() < deadline - 25:
                        _try_optimize_smooth(pts, maxiter=1200)
                        
                except Exception:
                    pass

    # 3. Explore four-ring configurations
    four_ring_splits = [(4, 3, 3, 4), (3, 4, 3, 4), (4, 4, 3, 3), (5, 3, 3, 3), (3, 3, 4, 4)]
    
    for n1, n2, n3, n4 in four_ring_splits:
        if time.time() > deadline - 25:
            break
        
        def _four_ring_score(params):
            r1, r2, r3, r4 = params[0], params[1], params[2], params[3]
            rot1, rot2, rot3, rot4 = params[4], params[5], params[6], params[7]
            
            if r1 <= 0.01 or r2 <= 0.01 or r3 <= 0.01 or r4 <= 0.01:
                return -1e6
                
            pts = _generate_four_ring_config(n1, n2, n3, n4, r1, r2, r3, r4, rot1, rot2, rot3, rot4)
            sc = _score(_normalize(pts))
            return -sc

        r_starts = [
            [1.0, 0.6, 0.35, 0.15],
            [1.0, 0.5, 0.3, 0.1]
        ]
        
        for r_start in r_starts:
            if time.time() > deadline - 25:
                break
            
            rot_start = [0.0, np.pi/n2, 0.0, np.pi/n4]
            x0 = np.array(r_start + rot_start)
            try:
                res = scipy.optimize.minimize(
                    _four_ring_score,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                r1, r2, r3, r4, rot1, rot2, rot3, rot4 = res.x
                if r1 <= 0.01 or r2 <= 0.01 or r3 <= 0.01 or r4 <= 0.01:
                    continue
                pts = _generate_four_ring_config(n1, n2, n3, n4, r1, r2, r3, r4, rot1, rot2, rot3, rot4)
                sc = _update_best(pts)
                
                if sc > best_score * 0.85 and time.time() < deadline - 20:
                    _try_optimize_smooth(pts, maxiter=1000)
                    
            except Exception:
                pass

    # 4. Random restarts with smooth optimization
    if time.time() < deadline - 20:
        for _ in range(4):
            if time.time() > deadline - 10:
                break
            
            angles = np.sort(rng.uniform(0, 2 * np.pi, N))
            radii = rng.uniform(0.4, 1.0, N)
            pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            
            sc = _update_best(pts)
            if sc > best_score * 0.5 and time.time() < deadline - 12:
                _try_optimize_smooth(pts, maxiter=1500)

    # Final: if we have time, do one more polish on the best points
    if best_points is not None and time.time() < deadline - 5:
        _try_optimize_smooth(best_points, beta_start=100.0, beta_end=3000.0, maxiter=2000)

    if best_points is None:
        best_points = _normalize(_regular_polygon(14, 1.0, 0.0))
    else:
        best_points = _normalize(best_points)
        
    return best_points
