import numpy as np
import scipy.spatial
import scipy.optimize
import time
from itertools import combinations

N = 14

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
        ha = float(scipy.spatial.ConvexHull(points).volume)
        return ha
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

def _neg_score(flat):
    return -_score(flat.reshape(-1, 2))

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

def _two_ring_config(n, frac=0.5, phase=0.5):
    n_outer = n - 7
    n_inner = 7
    outer = _regular_polygon(n_outer, 1.0, 0.0)
    inner = _regular_polygon(n_inner, frac, phase * 2 * np.pi / n_inner)
    return np.vstack([outer, inner])

def _ellipse_config(n, a=1.0, b=0.8, rotation=0.0):
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    x = a * np.cos(theta)
    y = b * np.sin(theta)
    pts = np.column_stack([x, y])
    c, s = np.cos(rotation), np.sin(rotation)
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T

def _build_structured_starts():
    starts = []
    for rot in [0.0, np.pi / N, np.pi / 14, np.pi / 7]:
        starts.append(_regular_polygon(N, 1.0, rot))
    
    rng = np.random.RandomState(42)
    for _ in range(5):
        base = _regular_polygon(N, 1.0, 0.0)
        jitter = rng.normal(0, 0.05, (N, 2))
        starts.append(base + jitter)
        
    for frac in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for phase in [0.0, 0.25, 0.5]:
            starts.append(_two_ring_config(N, frac, phase))
            
    for a in [0.9, 1.0, 1.1]:
        for b in [0.7, 0.8, 0.9, 1.0]:
            starts.append(_ellipse_config(N, a, b, 0.0))
            starts.append(_ellipse_config(N, a, b, np.pi/6))
            
    return starts

def _smooth_neg_score_and_grad(flat, beta=100.0):
    """
    Smooth objective with gradient using log-sum-exp approximation of min triangle area.
    """
    pts = flat.reshape(-1, 2)
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 1e10, np.zeros_like(flat)
    
    pi = pts[_I]
    pj = pts[_J]
    pk = pts[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    max_area = np.max(areas)
    if max_area < 1e-15:
        return 1e10, np.zeros_like(flat)
    
    shifted = areas - max_area
    exp_vals = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_vals)
    if sum_exp == 0:
        return 1e10, np.zeros_like(flat)
    weights = exp_vals / sum_exp
    
    log_sum = -beta * max_area + np.log(sum_exp)
    smooth_min = -log_sum / beta
    
    score = smooth_min / ha
    
    signs = np.sign(cross)
    
    d_cross_di_x = -(v1[:, 1] + v2[:, 1])
    d_cross_di_y = (v1[:, 0] + v2[:, 0])
    d_cross_dj_x = v2[:, 1]
    d_cross_dj_y = -v2[:, 0]
    d_cross_dk_x = -v1[:, 1]
    d_cross_dk_y = v1[:, 0]
    
    w = weights * signs * 0.5
    
    grad_points = np.zeros_like(pts)
    
    dA_di = w[:, None] * np.column_stack([d_cross_di_x, d_cross_di_y])
    dA_dj = w[:, None] * np.column_stack([d_cross_dj_x, d_cross_dj_y])
    dA_dk = w[:, None] * np.column_stack([d_cross_dk_x, d_cross_dk_y])
    
    np.add.at(grad_points, _I, dA_di)
    np.add.at(grad_points, _J, dA_dj)
    np.add.at(grad_points, _K, dA_dk)
    
    grad_score = grad_points / ha
    
    return -score, -grad_score.flatten()

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill

    def _time_left():
        return deadline - time.time()

    starts = _build_structured_starts()
    scored = []
    for s in starts:
        if _time_left() <= 2.0:
            break
        sn = _normalize(s)
        scored.append((_score(sn), sn))
    scored.sort(key=lambda x: -x[0])

    best_score = 0.0
    best_pts = _normalize(_regular_polygon(N, 1.0, 0.0))
    if scored:
        s0, p0 = scored[0]
        best_score, best_pts = s0, p0.copy()

    top_candidates = [pts for _, pts in scored[:10]]

    def _update_best(points):
        nonlocal best_score, best_pts
        pts = _normalize(points)
        sc = _score(pts)
        if sc > best_score:
            best_score, best_pts = sc, pts.copy()
        return sc

    def _try_optimize_smooth(init_points, beta_start=50.0, beta_end=500.0, maxiter=2000):
        if _time_left() <= 5.0:
            return
        init_norm = _normalize(init_points)
        x0 = init_norm.flatten()
        
        n_stages = 5
        betas = np.linspace(beta_start, beta_end, n_stages)
        
        x = x0.copy()
        for beta in betas:
            if _time_left() <= 5.0:
                break
            
            def _obj_and_grad(x_flat, beta_val=beta):
                return _smooth_neg_score_and_grad(x_flat, beta_val)
            
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
        
        if _time_left() > 3.0:
            try:
                def _obj_and_grad_high_beta(x_flat):
                    return _smooth_neg_score_and_grad(x_flat, 1000.0)
                
                res = scipy.optimize.minimize(
                    _obj_and_grad_high_beta,
                    x,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-14}
                )
                x = res.x
                _update_best(x.reshape(-1, 2))
                
                if _time_left() > 1.0:
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

    # Phase 1: Smooth optimization on top candidates
    for pts in top_candidates:
        if _time_left() <= 15.0:
            break
        _try_optimize_smooth(pts)

    # Phase 2: Refine best with tighter beta
    if _time_left() > 20.0:
        for beta in [200, 500, 1000]:
            if _time_left() <= 10.0:
                break
            def _obj_and_grad(x_flat, beta_val=beta):
                return _smooth_neg_score_and_grad(x_flat, beta_val)
            try:
                res = scipy.optimize.minimize(
                    _obj_and_grad,
                    best_pts.flatten(),
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'gtol': 1e-10, 'ftol': 1e-12}
                )
                _update_best(res.x.reshape(-1, 2))
            except Exception:
                pass
            if _time_left() > 5.0:
                try:
                    res = scipy.optimize.minimize(
                        _neg_score,
                        best_pts.flatten(),
                        method='Nelder-Mead',
                        options={'maxiter': 2000, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True}
                    )
                    _update_best(res.x.reshape(-1, 2))
                except Exception:
                    pass

    # Phase 3: Powell on best
    if _time_left() > 15.0:
        try:
            res = scipy.optimize.minimize(
                _neg_score,
                best_pts.flatten(),
                method='Powell',
                options={'maxiter': 3000, 'ftol': 1e-13, 'xtol': 1e-11, 'adaptive': True}
            )
            _update_best(res.x.reshape(-1, 2))
        except Exception:
            pass

    # Phase 4: Final Nelder-Mead polish
    if _time_left() > 8.0:
        try:
            res = scipy.optimize.minimize(
                _neg_score,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
            )
            _update_best(res.x.reshape(-1, 2))
        except Exception:
            pass

    # Phase 5: Ultra-tight polish
    if _time_left() > 4.0:
        try:
            res = scipy.optimize.minimize(
                _neg_score,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            _update_best(res.x.reshape(-1, 2))
        except Exception:
            pass

    return _normalize(best_pts)
