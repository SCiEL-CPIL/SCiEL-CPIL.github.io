import numpy as np
import scipy.spatial
import scipy.optimize
import time
from itertools import combinations

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
_M = len(_TRI_INDICES)

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

def _smooth_min_triangle_area(points, beta=100.0):
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
    log_sum = -beta * max_area + np.log(np.sum(exp_vals))
    smooth_min = -log_sum / beta
    return smooth_min

def _smooth_score(points, beta=100.0):
    try:
        if len(points) < 3:
            return 0.0
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
        smta = _smooth_min_triangle_area(points, beta)
        return smta / ha
    except Exception:
        return 0.0

def _neg_smooth_score_and_grad(points_flat, beta=100.0):
    """
    Returns (-smooth_score, gradient) for L-BFGS-B.
    Uses analytic gradient of the log-sum-exp soft-min.
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
    
    d_cross_di_x = -(v1[:, 1] + v2[:, 1])
    d_cross_di_y = (v1[:, 0] + v2[:, 0])
    d_cross_dj_x = v2[:, 1]
    d_cross_dj_y = -v2[:, 0]
    d_cross_dk_x = -v1[:, 1]
    d_cross_dk_y = v1[:, 0]
    
    # d(A_k)/d(P) = 0.5 * sign(cross_k) * d(cross_k)/d(P)
    # But since we're using |cross|, the gradient of 0.5*|cross| is:
    # If cross > 0: 0.5 * d(cross)/d(P)
    # If cross < 0: -0.5 * d(cross)/d(P)
    # So d(A)/d(P) = 0.5 * sign(cross) * d(cross)/d(P)
    
    w = weights * signs * 0.5
    
    grad_points = np.zeros_like(points)
    
    dA_di = w[:, None] * np.column_stack([d_cross_di_x, d_cross_di_y])
    dA_dj = w[:, None] * np.column_stack([d_cross_dj_x, d_cross_dj_y])
    dA_dk = w[:, None] * np.column_stack([d_cross_dk_x, d_cross_dk_y])
    
    np.add.at(grad_points, _I, dA_di)
    np.add.at(grad_points, _J, dA_dj)
    np.add.at(grad_points, _K, dA_dk)
    
    # Gradient of score = smooth_min / ha
    # We approximate grad_score ≈ grad_smooth_min / ha
    # This is valid when the hull is stable (all points remain on hull)
    
    grad_score = grad_points / ha
    
    return -score, -grad_score.flatten()

def _exact_score_and_grad(points_flat):
    """
    Compute exact min triangle area score and its subgradient.
    The min is achieved by one or more triangles. We use the subgradient
    from the active (minimum) triangles.
    """
    points = points_flat.reshape(-1, 2)
    
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    if min_area < 1e-15:
        return 0.0, np.zeros_like(points_flat)
    
    ha = _hull_area(points)
    if ha < 1e-12:
        return 0.0, np.zeros_like(points_flat)
    
    score = min_area / ha
    
    # Find all triangles achieving (near) the minimum
    tol = 1e-10 * min_area
    active = np.where(areas <= min_area + tol)[0]
    
    # Subgradient: average of gradients of active triangles
    signs = np.sign(cross[active])
    
    d_cross_di_x = -(v1[active, 1] + v2[active, 1])
    d_cross_di_y = (v1[active, 0] + v2[active, 0])
    d_cross_dj_x = v2[active, 1]
    d_cross_dj_y = -v2[active, 0]
    d_cross_dk_x = -v1[active, 1]
    d_cross_dk_y = v1[active, 0]
    
    w = signs * 0.5 / len(active)
    
    grad_points = np.zeros_like(points)
    
    dA_di = w[:, None] * np.column_stack([d_cross_di_x, d_cross_di_y])
    dA_dj = w[:, None] * np.column_stack([d_cross_dj_x, d_cross_dj_y])
    dA_dk = w[:, None] * np.column_stack([d_cross_dk_x, d_cross_dk_y])
    
    np.add.at(grad_points, _I[active], dA_di)
    np.add.at(grad_points, _J[active], dA_dj)
    np.add.at(grad_points, _K[active], dA_dk)
    
    grad_score = grad_points / ha
    
    return score, grad_score.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_score = 0.0
    best_points = None

    def _update_best(points):
        nonlocal best_score, best_points
        pts = _normalize(points)
        sc = _score(pts)
        if sc > best_score:
            best_score = sc
            best_points = pts.copy()
        return sc

    def _try_optimize_smooth(init_points, beta_start=30.0, beta_end=300.0, n_stages=4, maxiter_per_stage=800):
        """
        Optimize using smooth surrogate with annealed beta.
        """
        if time.time() > deadline - 10:
            return
        init_norm = _normalize(init_points)
        x0 = init_norm.flatten()
        
        betas = np.linspace(beta_start, beta_end, n_stages)
        
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
                    options={'maxiter': maxiter_per_stage, 'gtol': 1e-9, 'ftol': 1e-13}
                )
                x = res.x
                _update_best(x.reshape(-1, 2))
            except Exception:
                pass
        
        # Final polish with high beta
        if time.time() < deadline - 5:
            try:
                def _obj_and_grad_high_beta(x_flat):
                    return _neg_smooth_score_and_grad(x_flat, 800.0)
                
                res = scipy.optimize.minimize(
                    _obj_and_grad_high_beta,
                    x,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'gtol': 1e-10, 'ftol': 1e-14}
                )
                x = res.x
                _update_best(x.reshape(-1, 2))
            except Exception:
                pass

    def _try_subgradient_polish(init_points, n_iter=200, step_size=0.01):
        """
        Use subgradient method on the exact objective for final polish.
        """
        if time.time() > deadline - 5:
            return
        
        x = _normalize(init_points).flatten()
        
        for i in range(n_iter):
            if time.time() > deadline - 3:
                break
            try:
                score, grad = _exact_score_and_grad(x)
                if grad is None or np.linalg.norm(grad) < 1e-15:
                    break
                # Adaptive step size
                ss = step_size / (1.0 + i * 0.01)
                x_new = x + ss * grad
                sc_new = _score(x_new.reshape(-1, 2))
                if sc_new > _score(x.reshape(-1, 2)):
                    x = x_new
                    _update_best(x.reshape(-1, 2))
                else:
                    step_size *= 0.5
            except Exception:
                break

    # Strategy 1: Two-ring configurations with full coordinate optimization
    splits = [(7, 7), (6, 8), (8, 6), (5, 9), (9, 5), (4, 10), (10, 4), (3, 11), (11, 3)]
    
    for n1, n2 in splits:
        if time.time() > deadline - 20:
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

        for r_init in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            for rot_init in [0.0, np.pi/n1, np.pi/n2, np.pi/(n1+n2)]:
                if time.time() > deadline - 15:
                    continue
                x0 = np.array([r_init, 0.0, rot_init])
                try:
                    res = scipy.optimize.minimize(
                        _two_ring_score,
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 400, 'xatol': 1e-9, 'fatol': 1e-11}
                    )
                    r_ratio, rot1, rot2 = res.x
                    pts = _generate_two_ring_config(n1, n2, 1.0, r_ratio, rot1, rot2)
                    sc = _update_best(pts)
                    
                    if sc > best_score * 0.90 and time.time() < deadline - 25:
                        _try_optimize_smooth(pts, beta_start=30.0, beta_end=300.0, n_stages=4, maxiter_per_stage=600)
                        
                except Exception:
                    pass

    # Strategy 2: Three-ring configurations
    three_ring_splits = [(5, 4, 5), (6, 4, 4), (7, 3, 4), (4, 5, 5), (5, 5, 4), (6, 5, 3)]
    
    for n1, n2, n3 in three_ring_splits:
        if time.time() > deadline - 15:
            break
            
        def _three_ring_score(params):
            r2_ratio = params[0]
            r3_ratio = params[1]
            rot2 = params[2]
            rot3 = params[3]
            
            r1 = 1.0
            r2 = r2_ratio * r1
            r3 = r3_ratio * r1
            
            if r2 <= 0.01 or r3 <= 0.01:
                return -1e6
                
            pts = _generate_three_ring_config(n1, n2, n3, r1, r2, r3, 0.0, rot2, rot3)
            sc = _score(_normalize(pts))
            return -sc

        for r2_init in [0.4, 0.5, 0.6]:
            for r3_init in [0.2, 0.3, 0.4]:
                if r3_init >= r2_init:
                    continue
                for rot2_init in [0.0, np.pi/n2]:
                    for rot3_init in [0.0, np.pi/n3]:
                        if time.time() > deadline - 10:
                            continue
                        x0 = np.array([r2_init, r3_init, rot2_init, rot3_init])
                        try:
                            res = scipy.optimize.minimize(
                                _three_ring_score,
                                x0,
                                method='Nelder-Mead',
                                options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-11}
                            )
                            r2_ratio, r3_ratio, rot2, rot3 = res.x
                            pts = _generate_three_ring_config(n1, n2, n3, 1.0, r2_ratio, r3_ratio, 0.0, rot2, rot3)
                            sc = _update_best(pts)
                            
                            if sc > best_score * 0.90 and time.time() < deadline - 20:
                                _try_optimize_smooth(pts, beta_start=30.0, beta_end=250.0, n_stages=3, maxiter_per_stage=500)
                                
                        except Exception:
                            pass

    # Strategy 3: Regular polygon and perturbations
    reg = _regular_polygon(N, 1.0)
    _update_best(reg)
    
    rng = np.random.RandomState(42)
    for i in range(12):
        if time.time() > deadline - 15:
            break
        noise = rng.uniform(-0.08, 0.08, (N, 2))
        init = _regular_polygon(N, 1.0) + noise
        _try_optimize_smooth(init, beta_start=25.0, beta_end=250.0, n_stages=3, maxiter_per_stage=500)

    # Strategy 4: Random restarts
    for i in range(6):
        if time.time() > deadline - 12:
            break
        r = np.sqrt(rng.uniform(0, 1, N))
        theta = rng.uniform(0, 2 * np.pi, N)
        init = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        _try_optimize_smooth(init, beta_start=20.0, beta_end=200.0, n_stages=3, maxiter_per_stage=400)

    # Strategy 5: Subgradient polish on best found
    if best_points is not None and time.time() < deadline - 5:
        _try_subgradient_polish(best_points, n_iter=150, step_size=0.005)

    # Final check
    if best_points is None:
        best_points = _normalize(_regular_polygon(N, 1.0))
    else:
        best_points = _normalize(best_points)
        
    return best_points
