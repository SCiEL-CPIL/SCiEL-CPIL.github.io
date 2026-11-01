import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14
DEADLINE = 135.0  # wall-clock guard (15s safety margin before 150s kill)

def _triangle_indices(n):
    idx = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                idx.append((i, j, k))
    return np.array(idx, dtype=np.int32)

_TRI = _triangle_indices(N)
_I = _TRI[:, 0]
_J = _TRI[:, 1]
_K = _TRI[:, 2]
_NTRI = _TRI.shape[0]

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
        return float(scipy.spatial.ConvexHull(points).volume)
    except Exception:
        return 0.0

def _score(points):
    try:
        if len(points) < 3:
            return 0.0
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
        return _min_triangle_area(points) / ha
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
    a = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    return np.column_stack([radius * np.cos(a), radius * np.sin(a)])

def _two_ring_config(n_outer, n_inner, r_inner, phase=0.0):
    outer = _regular_polygon(n_outer, 1.0, 0.0)
    inner = _regular_polygon(n_inner, r_inner, phase)
    return np.vstack([outer, inner])

def _ellipse_config(n, a=1.0, b=0.8, rotation=0.0):
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    x = a * np.cos(theta)
    y = b * np.sin(theta)
    pts = np.column_stack([x, y])
    c, s = np.cos(rotation), np.sin(rotation)
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T

def _smooth_min_area(points, beta):
    """Log-sum-exp smoothed minimum of triangle areas."""
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    max_a = np.max(areas)
    if max_a < 1e-15:
        return 0.0, np.zeros_like(areas)
    
    shifted = areas - max_a
    exp_vals = np.exp(-beta * shifted)
    log_sum = np.log(np.sum(exp_vals))
    smooth_min = max_a + log_sum / beta  # = -(1/beta)*log(sum(exp(-beta*(areas - max_a)))) + max_a
    # Actually: log-sum-exp of (-beta * areas) = -beta*max_a + log(sum(exp(-beta*(areas - max_a))))
    # So min approximation = -1/beta * LSE(-beta*areas) = max_a - log(sum(exp(-beta*(areas-max_a))))/beta
    # Wait, let me re-derive:
    # LSE(x) = log(sum(exp(x)))
    # We want to approximate min(areas) = -1/beta * log(sum(exp(-beta*areas)))
    # = -1/beta * [ -beta*max_a + log(sum(exp(-beta*(areas - max_a)))) ]
    # = max_a - (1/beta)*log(sum(exp(-beta*(areas - max_a))))
    # Since shifted = areas - max_a >= 0, exp(-beta*shifted) <= 1
    # log(sum(exp(-beta*shifted))) <= log(NTRI)
    # So smooth_min = max_a - (1/beta)*log_sum where log_sum = log(sum(exp(-beta*shifted)))
    
    smooth_min = max_a - log_sum / beta
    weights = exp_vals / np.sum(exp_vals)
    return smooth_min, weights

def _neg_smooth_score_and_grad(flat, beta):
    """Returns (-score, -gradient) for L-BFGS-B with analytic gradient."""
    points = flat.reshape(-1, 2)
    
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    max_a = np.max(areas)
    if max_a < 1e-15:
        return 1e10, np.zeros_like(flat)
    
    shifted = areas - max_a
    exp_vals = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_vals)
    if sum_exp < 1e-300:
        return 1e10, np.zeros_like(flat)
    
    weights = exp_vals / sum_exp
    log_sum = np.log(sum_exp)
    smooth_min = max_a - log_sum / beta
    
    ha = _hull_area(points)
    if ha < 1e-12:
        return 1e10, np.zeros_like(flat)
    
    score = smooth_min / ha
    
    # Gradient of smooth_min w.r.t. points
    # smooth_min = max_a - (1/beta)*log(sum(exp(-beta*(areas - max_a))))
    # d(smooth_min)/d(points) = -(1/beta) * d/d(points) log(sum(exp(-beta*(areas - max_a))))
    # = -(1/beta) * sum_k [ exp(-beta*(areas_k - max_a)) / sum_j exp(-beta*(areas_j - max_a)) * (-beta) * d(areas_k)/d(points) ]
    # = sum_k [ weights_k * d(areas_k)/d(points) ]
    # But wait: areas = 0.5*|cross|, so d(areas)/d(points) = 0.5 * sign(cross) * d(cross)/d(points)
    
    signs = np.sign(cross)
    w = weights * signs * 0.5  # weights * d(areas_k)/d(cross_k)
    
    # d(cross)/d(point_i) where cross = (pj-pi)x(pk-pi) = v1 x v2
    # v1 = pj - pi, v2 = pk - pi
    # cross = v1[0]*v2[1] - v1[1]*v2[0]
    # d(cross)/d(pi[0]) = -v2[1] - (-v1[1]) ... let me be careful
    # cross = (pj[0]-pi[0])*(pk[1]-pi[1]) - (pj[1]-pi[1])*(pk[0]-pi[0])
    # d/d(pi[0]) = -(pk[1]-pi[1]) + (pj[1]-pi[1]) = -v2[1] + v1[1] = v1[1] - v2[1]
    # d/d(pi[1]) = (pj[0]-pi[0]) - (pk[0]-pi[0]) = v1[0] - v2[0]
    # d/d(pj[0]) = (pk[1]-pi[1]) = v2[1]
    # d/d(pj[1]) = -(pk[0]-pi[0]) = -v2[0]
    # d/d(pk[0]) = -(pj[1]-pi[1]) = -v1[1]
    # d/d(pk[1]) = (pj[0]-pi[0]) = v1[0]
    
    d_cross_di_x = v1[:, 1] - v2[:, 1]
    d_cross_di_y = v1[:, 0] - v2[:, 0]
    d_cross_dj_x = v2[:, 1]
    d_cross_dj_y = -v2[:, 0]
    d_cross_dk_x = -v1[:, 1]
    d_cross_dk_y = v1[:, 0]
    
    grad_points = np.zeros_like(points)
    
    dA_di = w[:, None] * np.column_stack([d_cross_di_x, d_cross_di_y])
    dA_dj = w[:, None] * np.column_stack([d_cross_dj_x, d_cross_dj_y])
    dA_dk = w[:, None] * np.column_stack([d_cross_dk_x, d_cross_dk_y])
    
    np.add.at(grad_points, _I, dA_di)
    np.add.at(grad_points, _J, dA_dj)
    np.add.at(grad_points, _K, dA_dk)
    
    # d(score)/d(points) = grad_points / ha
    # We want to minimize -score, so gradient of -score = -grad_points/ha
    grad_score = grad_points / ha
    neg_grad = -grad_score.flatten()
    
    return -score, neg_grad

def _build_structured_starts():
    starts = []
    # Regular polygon baseline
    for rot in [0.0, np.pi / N, np.pi / 14, np.pi / 3, np.pi / 7]:
        starts.append(_regular_polygon(N, 1.0, rot))
    
    # Two-ring configurations with varied splits
    for n_outer in [7, 8, 9, 10]:
        n_inner = N - n_outer
        for frac in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            for phase in [0.0, 0.25, 0.5, 0.75]:
                starts.append(_two_ring_config(n_outer, n_inner, frac, phase * 2 * np.pi / n_inner))
                
    # Elliptical hulls
    for a in [0.9, 1.0, 1.1]:
        for b in [0.7, 0.8, 0.9, 1.0]:
            for rot in [0.0, np.pi/6, np.pi/4, np.pi/3]:
                starts.append(_ellipse_config(N, a, b, rot))
                
    # Three-ring configuration
    n1, n2, n3 = 5, 5, 4
    for r1 in [0.3, 0.4]:
        for r2 in [0.6, 0.7]:
            ring1 = _regular_polygon(n1, 1.0, 0.0)
            ring2 = _regular_polygon(n2, r2, np.pi/n2)
            ring3 = _regular_polygon(n3, r1, np.pi/n3)
            starts.append(np.vstack([ring1, ring2, ring3]))
            
    return starts

def construct():
    t0 = time.time()
    def left():
        return DEADLINE - (time.time() - t0)

    starts = _build_structured_starts()
    scored = []
    for s in starts:
        if left() <= 2.0:
            break
        sn = _normalize(s)
        scored.append((_score(sn), sn))
    scored.sort(key=lambda x: -x[0])

    best_score, best_pts = 0.0, _normalize(_regular_polygon(N, 1.0, 0.0))
    if scored:
        s0, p0 = scored[0]
        best_score, best_pts = s0, p0.copy()

    top_candidates = [pts for _, pts in scored[:8]]

    def update_best(points):
        nonlocal best_score, best_pts
        try:
            opt = _normalize(points)
            sc = _score(opt)
            if sc > best_score:
                best_score, best_pts = sc, opt.copy()
        except Exception:
            pass

    def try_smooth_lbfgs(x0, beta_schedule, maxiter_per_stage):
        """Multi-stage L-BFGS-B with increasing beta (sharper min approximation)."""
        nonlocal best_score, best_pts
        x = x0.copy()
        for beta in beta_schedule:
            if left() <= 5.0:
                break
            def obj_grad(f, b=beta):
                return _neg_smooth_score_and_grad(f, b)
            try:
                res = scipy.optimize.minimize(
                    obj_grad, x, jac=True, method='L-BFGS-B',
                    options={'maxiter': maxiter_per_stage, 'gtol': 1e-8, 'ftol': 1e-14, 'maxls': 50}
                )
                x = res.x
                update_best(x.reshape(-1, 2))
            except Exception:
                pass
        return x

    def try_nm(flat, maxiter, fatol, xatol):
        nonlocal best_score, best_pts
        if left() <= 4.0:
            return
        try:
            def neg_score(f):
                return -_score(f.reshape(-1, 2))
            res = scipy.optimize.minimize(
                neg_score, flat, method='Nelder-Mead',
                options={'maxiter': maxiter, 'fatol': fatol, 'xatol': xatol, 'adaptive': True}
            )
            update_best(res.x.reshape(-1, 2))
        except Exception:
            pass

    # Phase 1: Nelder-Mead on top candidates (explore local landscape)
    for pts in top_candidates[:4]:
        if left() <= 15.0:
            break
        try_nm(pts.flatten(), 5000, 1e-13, 1e-10)

    # Phase 2: Multi-stage smooth L-BFGS-B from best (refine with analytic gradients)
    if left() > 30.0:
        beta_sched = [20.0, 50.0, 100.0, 200.0, 400.0, 800.0]
        try_smooth_lbfgs(best_pts.flatten(), beta_sched, 300)

    # Phase 3: Powell on best (derivative-free, good for non-smooth)
    if left() > 20.0:
        try:
            def neg_score(f):
                return -_score(f.reshape(-1, 2))
            res = scipy.optimize.minimize(
                neg_score, best_pts.flatten(), method='Powell',
                options={'maxiter': 3000, 'fatol': 1e-14, 'xatol': 1e-12}
            )
            update_best(res.x.reshape(-1, 2))
        except Exception:
            pass

    # Phase 4: More smooth L-BFGS-B with very high beta (near-exact min)
    if left() > 15.0:
        try_smooth_lbfgs(best_pts.flatten(), [1000.0, 2000.0], 200)

    # Phase 5: Final Nelder-Mead polish
    if left() > 6.0:
        try_nm(best_pts.flatten(), 3000, 1e-15, 1e-13)

    # Phase 6: One more round of smooth refinement from current best
    if left() > 8.0:
        try_smooth_lbfgs(best_pts.flatten(), [500.0, 1500.0, 3000.0], 150)

    # Final NM polish
    if left() > 3.0:
        try_nm(best_pts.flatten(), 1000, 1e-15, 1e-14)

    return _normalize(best_pts)
