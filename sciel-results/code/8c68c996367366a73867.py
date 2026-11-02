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

def _neg_smooth_score_and_grad(points_flat, beta=100.0, repulsion=0.01):
    points = points_flat.reshape(-1, 2)
    
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft-min using log-sum-exp
    max_area = np.max(areas)
    if max_area < 1e-15:
        return 0.0, np.zeros_like(points_flat)

    shifted = areas - max_area
    # Clip to avoid overflow
    shifted = np.clip(shifted, -100, 0)
    exp_vals = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_vals)
    if sum_exp < 1e-300:
        return 0.0, np.zeros_like(points_flat)
    weights = exp_vals / sum_exp
    
    log_sum = -beta * max_area + np.log(sum_exp)
    smooth_min = -log_sum / beta
    
    ha = _hull_area(points)
    if ha < 1e-12:
        return 0.0, np.zeros_like(points_flat)
    
    score = smooth_min / ha
    
    # Add repulsion term to prevent point collisions
    # Compute pairwise distances
    n = len(points)
    repulsion_score = 0.0
    repulsion_grad = np.zeros_like(points)
    
    if repulsion > 0:
        for i in range(n):
            for j in range(i+1, n):
                d = points[j] - points[i]
                dist = np.linalg.norm(d)
                if dist < 0.01:
                    # Strong repulsion when very close
                    repulsion_score += 1.0 / (dist + 1e-10)
                    repulsion_grad[i] -= d / (dist * dist + 1e-20)
                    repulsion_grad[j] += d / (dist * dist + 1e-20)
    
    total_score = score - repulsion * repulsion_score
    
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
    
    grad_score = grad_points / ha - repulsion * repulsion_grad
    
    return -total_score, -grad_score.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # 2s safety margin before 150s kill

    best_score = 0.0
    best_points = None
    rng = np.random.RandomState(42)

    def _update_best(points):
        nonlocal best_score, best_points
        if points is None:
            return 0.0
        pts = _normalize(points)
        sc = _score(pts)
        if sc > best_score:
            best_score = sc
            best_points = pts.copy()
        return sc

    def _try_optimize_smooth(init_points, beta_start=20.0, beta_end=2000.0, maxiter=2000):
        if time.time() > deadline - 3:
            return
        init_norm = _normalize(init_points)
        x0 = init_norm.flatten()
        
        n_stages = 8
        betas = np.logspace(np.log10(beta_start), np.log10(beta_end), n_stages)
        
        x = x0.copy()
        for i, beta in enumerate(betas):
            if time.time() > deadline - 5:
                break
            
            def _obj_and_grad(x_flat, beta_val=beta):
                return _neg_smooth_score_and_grad(x_flat, beta_val, repulsion=0.005)
            
            try:
                # Use SLSQP for better gradient handling in early stages
                if i < n_stages // 2:
                    res = scipy.optimize.minimize(
                        _obj_and_grad,
                        x,
                        jac=True,
                        method='SLSQP',
                        options={'maxiter': maxiter // n_stages, 'ftol': 1e-12}
                    )
                else:
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
        
        # Final high-precision polish without repulsion
        if time.time() < deadline - 3:
            try:
                def _obj_and_grad_high_beta(x_flat):
                    return _neg_smooth_score_and_grad(x_flat, 3000.0, repulsion=0.0)
                
                res = scipy.optimize.minimize(
                    _obj_and_grad_high_beta,
                    x,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'gtol': 1e-12, 'ftol': 1e-15}
                )
                x = res.x
                _update_best(x.reshape(-1, 2))
                
                if time.time() < deadline - 1:
                    # Exact score optimization with Nelder-Mead
                    def _neg_exact_score(x_flat):
                        return -_score(x_flat.reshape(-1, 2))
                    
                    res = scipy.optimize.minimize(
                        _neg_exact_score,
                        x,
                        method='Nelder-Mead',
                        options={'maxiter': 500, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
                    )
                    _update_best(res.x.reshape(-1, 2))
            except Exception:
                pass

    # 1. Explore the two-ring family analytically
    splits = [(7, 7), (6, 8), (8, 6), (5, 9), (9, 5), (4, 10), (10, 4), (3, 11), (11, 3), (2, 12), (12, 2)]
    
    two_ring_candidates = []
    
    for n1, n2 in splits:
        if time.time() > deadline - 25:
            break
            
        def _two_ring_score(params, n1_val=n1, n2_val=n2):
            r_ratio = params[0]
            rot1 = params[1]
            rot2 = params[2]
            
            r1 = 1.0
            r2 = r_ratio * r1
            
            if r2 <= 0.01 or r2 >= 0.99:
                return 1e6
                
            pts = _generate_two_ring_config(n1_val, n2_val, r1, r2, rot1, rot2)
            sc = _score(_normalize(pts))
            return -sc

        found_local_best = False
        best_local_score = -1e10
        best_local_pts = None
        
        # More diverse initializations
        r_inits = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        rot_inits = [0.0, np.pi/(2*n1), np.pi/n1, np.pi/(2*n2), np.pi/n2]
        
        for r_init in r_inits:
            if time.time() > deadline - 10:
                break
            for rot_init in rot_inits:
                if time.time() > deadline - 5:
                    break
                x0 = np.array([r_init, rot_init, rot_init + np.pi/(n1+n2)])
                try:
                    res = scipy.optimize.minimize(
                        _two_ring_score,
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 800, 'xatol': 1e-9, 'fatol': 1e-11}
                    )
                    r_ratio, rot1, rot2 = res.x
                    pts = _generate_two_ring_config(n1, n2, 1.0, r_ratio, rot1, rot2)
                    sc = _score(_normalize(pts))
                    
                    if sc > best_local_score:
                        best_local_score = sc
                        best_local_pts = pts.copy()
                        found_local_best = True
                        
                except Exception:
                    pass
        
        if found_local_best and best_local_score > 0:
            two_ring_candidates.append((best_local_score, best_local_pts))

    # Sort candidates by score
    two_ring_candidates.sort(key=lambda x: x[0], reverse=True)

    # 2. Refine the top candidates with coordinate optimization
    top_k = 8
    for sc, pts in two_ring_candidates[:top_k]:
        if time.time() > deadline - 20:
            break
        if sc > best_score * 0.80 or best_score == 0:
            _try_optimize_smooth(pts, maxiter=1500)

    # 3. Explore regular polygon and perturbations
    reg = _regular_polygon(N, 1.0)
    _update_best(reg)
    
    for i in range(15):
        if time.time() > deadline - 15:
            break
        # Different noise scales
        noise_scale = 0.05 + 0.05 * (i % 3)
        noise = rng.uniform(-noise_scale, noise_scale, (N, 2))
        init = _regular_polygon(N, 1.0) + noise
        _try_optimize_smooth(init, maxiter=1000)

    # 4. Random restarts with optimization
    for i in range(8):
        if time.time() > deadline - 12:
            break
        # Mix of uniform and ring-based random points
        if i % 2 == 0:
            r = np.sqrt(rng.uniform(0, 1, N))
            theta = rng.uniform(0, 2 * np.pi, N)
            init = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        else:
            # Two random rings
            n1 = rng.randint(4, 11)
            n2 = N - n1
            r1 = rng.uniform(0.5, 0.9)
            r2 = rng.uniform(0.2, 0.4)
            rot1 = rng.uniform(0, 2*np.pi)
            rot2 = rng.uniform(0, 2*np.pi)
            init = _generate_two_ring_config(n1, n2, r1, r2, rot1, rot2)
        _try_optimize_smooth(init, maxiter=800)

    # 5. Coordinate Descent Polish with multiple passes
    if best_points is not None and time.time() < deadline - 8:
        pts = best_points.copy()
        n_cd_passes = 8
        for iteration in range(n_cd_passes):
            if time.time() > deadline - 3:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline - 2:
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
                        options={'maxiter': 150, 'xatol': 1e-12, 'fatol': 1e-14}
                    )
                    new_x = res.x
                    if np.linalg.norm(new_x - x0) > 1e-10:
                        improved = True
                    pts[i, 0] = new_x[0]
                    pts[i, 1] = new_x[1]
                except Exception:
                    pass
        _update_best(pts)
        
        # 6. Final high-precision polish with smooth objective
        if time.time() < deadline - 2:
            x = pts.flatten()
            for beta in [2000.0, 5000.0, 10000.0]:
                if time.time() > deadline - 0.5:
                    break
                try:
                    def _obj_and_grad_high(x_flat, b=beta):
                        return _neg_smooth_score_and_grad(x_flat, b, repulsion=0.0)
                    
                    res = scipy.optimize.minimize(
                        _obj_and_grad_high,
                        x,
                        jac=True,
                        method='L-BFGS-B',
                        options={'maxiter': 500, 'gtol': 1e-13, 'ftol': 1e-16}
                    )
                    x = res.x
                    _update_best(x.reshape(-1, 2))
                except Exception:
                    pass

    if best_points is None:
        best_points = _normalize(_regular_polygon(N, 1.0))
    else:
        best_points = _normalize(best_points)
        
    return best_points
