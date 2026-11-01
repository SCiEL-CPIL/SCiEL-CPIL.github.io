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

def _neg_smooth_score_and_grad(points_flat, beta=100.0):
    """
    Returns (-smooth_score, gradient) for L-BFGS-B.
    Uses analytic gradient of the smooth min triangle area / hull area.
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
    
    # Derivatives of cross product w.r.t. coordinates of points i, j, k
    # cross = (xj - xi)*(yk - yi) - (yj - yi)*(xk - xi)
    # d/dxi_x = -(yk - yi) = -(v1_y + v2_y)
    # d/dxi_y = (xj - xi) = (v1_x + v2_x)
    # d/dxj_x = (yk - yi) = v2_y
    # d/dxj_y = -(xk - xi) = -v2_x
    # d/dxk_x = -(yj - yi) = -v1_y
    # d/dxk_y = (xj - xi) = v1_x
    
    d_cross_di_x = -(v1[:, 1] + v2[:, 1])
    d_cross_di_y = (v1[:, 0] + v2[:, 0])
    d_cross_dj_x = v2[:, 1]
    d_cross_dj_y = -v2[:, 0]
    d_cross_dk_x = -v1[:, 1]
    d_cross_dk_y = v1[:, 0]
    
    # Gradient of 0.5 * |cross| is 0.5 * sign(cross) * d_cross
    # Weighted sum for smooth min
    w = weights * signs * 0.5
    
    grad_points = np.zeros_like(points)
    
    dA_di = w[:, None] * np.column_stack([d_cross_di_x, d_cross_di_y])
    dA_dj = w[:, None] * np.column_stack([d_cross_dj_x, d_cross_dj_y])
    dA_dk = w[:, None] * np.column_stack([d_cross_dk_x, d_cross_dk_y])
    
    np.add.at(grad_points, _I, dA_di)
    np.add.at(grad_points, _J, dA_dj)
    np.add.at(grad_points, _K, dA_dk)
    
    # Score = smooth_min / ha. Gradient = grad_smooth_min / ha (ignoring grad of ha for simplicity/stability in interior)
    # Actually, to be precise: d(score) = (d(smooth_min)*ha - smooth_min*d(ha))/ha^2
    # But d(ha) is complex. In the interior of the hull, moving points doesn't change the hull vertices usually, 
    # or if it does, it's a measure zero event. 
    # For optimization purposes, treating ha as constant (or approximating its gradient as 0) is standard 
    # for this type of max-min problem when the hull is stable. 
    # Given the points are likely on the hull, this might be tricky. 
    # However, the previous code used this approach and worked. 
    # Let's stick to grad_score = grad_points / ha
    
    grad_score = grad_points / ha
    
    return -score, -grad_score.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
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
        """
        Optimize using smooth surrogate with annealed beta.
        """
        if time.time() > deadline:
            return
        init_norm = _normalize(init_points)
        x0 = init_norm.flatten()
        
        # Anneal beta from beta_start to beta_end
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
        
        # Final polish with exact (non-smooth) score using Nelder-Mead
        if time.time() < deadline - 3:
            try:
                def _neg_exact_score(x_flat):
                    return -_score(x_flat.reshape(-1, 2))
                
                res = scipy.optimize.minimize(
                    _neg_exact_score,
                    x,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                _update_best(res.x.reshape(-1, 2))
            except Exception:
                pass

    # 1. Explore the two-ring family analytically
    # Focus heavily on (7,7) as per the strategy, but also check others.
    splits = [(7, 7), (6, 8), (8, 6)]
    
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

        # For (7,7), we can fix rot1=0 and optimize rot2 (relative rotation) and r_ratio.
        # But Nelder-Mead on 3 params is fast.
        # We will do a dense grid search for (7,7) first to find a good basin, then optimize.
        
        if n1 == 7 and n2 == 7:
            # Dense grid search for phi (rot2) and r_ratio
            # phi in [0, pi/7] due to symmetry
            # r_ratio in [0.2, 0.9]
            best_local_score = -1e6
            best_local_params = None
            
            r_inits = np.linspace(0.3, 0.8, 10)
            phi_inits = np.linspace(0, np.pi/7, 10)
            
            for r in r_inits:
                for phi in phi_inits:
                    if time.time() > deadline:
                        break
                    params = np.array([r, 0.0, phi])
                    sc = -_two_ring_score(params)
                    if sc > best_local_score:
                        best_local_score = sc
                        best_local_params = params.copy()
            
            if best_local_params is not None:
                # Refine the best grid point
                try:
                    res = scipy.optimize.minimize(
                        _two_ring_score,
                        best_local_params,
                        method='Nelder-Mead',
                        options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
                    )
                    r_ratio, rot1, rot2 = res.x
                    pts = _generate_two_ring_config(n1, n2, 1.0, r_ratio, rot1, rot2)
                    sc = _update_best(pts)
                    
                    # If good, refine with full coordinate optimization
                    if sc > best_score * 0.95 and time.time() < deadline - 20:
                        _try_optimize_smooth(pts, maxiter=3000)
                        
                except Exception:
                    pass
        else:
            # For other splits, just try a few random-ish starts
            for r_init in [0.4, 0.6, 0.8]:
                for rot_init in [0.0, np.pi/n1, np.pi/n2]:
                    if time.time() > deadline:
                        continue
                    x0 = np.array([r_init, 0.0, rot_init])
                    try:
                        res = scipy.optimize.minimize(
                            _two_ring_score,
                            x0,
                            method='Nelder-Mead',
                            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
                        )
                        r_ratio, rot1, rot2 = res.x
                        pts = _generate_two_ring_config(n1, n2, 1.0, r_ratio, rot1, rot2)
                        sc = _update_best(pts)
                        
                        if sc > best_score * 0.95 and time.time() < deadline - 15:
                            _try_optimize_smooth(pts, maxiter=2000)
                            
                    except Exception:
                        pass

    # 2. Explore regular polygon and slight perturbations
    reg = _regular_polygon(N, 1.0)
    _update_best(reg)
    
    # Perturb regular polygon and optimize with smooth surrogate
    for i in range(10):
        if time.time() > deadline - 10:
            break
        noise = rng.uniform(-0.1, 0.1, (N, 2))
        init = _regular_polygon(N, 1.0) + noise
        _try_optimize_smooth(init, maxiter=1500)

    # 3. Random restarts with optimization
    for i in range(5):
        if time.time() > deadline - 10:
            break
        r = np.sqrt(rng.uniform(0, 1, N))
        theta = rng.uniform(0, 2 * np.pi, N)
        init = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        _try_optimize_smooth(init, maxiter=1000)

    # Final check
    if best_points is None:
        best_points = _normalize(_regular_polygon(N, 1.0))
    else:
        best_points = _normalize(best_points)
        
    return best_points
