import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14
DEADLINE_FRAC = 145.0  # wall-clock guard (evaluator kills at 150s)
_HULL_CACHE = {}
_HULL_CACHE_MAX = 500

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
        key = tuple(np.round(points, 10).flatten())
        if key in _HULL_CACHE:
            return _HULL_CACHE[key]
        ha = float(scipy.spatial.ConvexHull(points).volume)
        if len(_HULL_CACHE) >= _HULL_CACHE_MAX:
            _HULL_CACHE.clear()
        _HULL_CACHE[key] = ha
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

def _build_structured_starts():
    starts = []
    for rot in [0.0, np.pi / N, np.pi / 14, np.pi / 7]:
        starts.append(_regular_polygon(N, 1.0, rot))
    
    for frac in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
        for phase in [0.0, 0.25, 0.5, 0.75]:
            starts.append(_two_ring_config(N, frac, phase))
            
    return starts

def _lse_min_area_and_grad(points, tau=0.01):
    """
    Smooth approximation of min triangle area using Log-Sum-Exp (soft-min)
    with analytical gradient.
    """
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_a = np.min(areas)
    exp_terms = np.exp(-(areas - min_a) / tau)
    sum_exp = np.sum(exp_terms)
    lse_val = -tau * np.log(sum_exp) + min_a
    
    weights = exp_terms / sum_exp
    
    n = len(points)
    
    sign_cross = np.sign(cross)
    
    d_pi0 = pj[:, 1] - pk[:, 1]
    d_pi1 = pk[:, 0] - pj[:, 0]
    d_pj0 = pk[:, 1] - pi[:, 1]
    d_pj1 = -(pk[:, 0] - pi[:, 0])
    d_pk0 = -(pj[:, 1] - pi[:, 1])
    d_pk1 = pj[:, 0] - pi[:, 0]
    
    grad_points = np.zeros((n, 2))
    np.add.at(grad_points, _I, weights[:, np.newaxis] * np.column_stack([d_pi0, d_pi1]) * 0.5)
    np.add.at(grad_points, _J, weights[:, np.newaxis] * np.column_stack([d_pj0, d_pj1]) * 0.5)
    np.add.at(grad_points, _K, weights[:, np.newaxis] * np.column_stack([d_pk0, d_pk1]) * 0.5)
    
    return lse_val, grad_points

def _smooth_neg_score_and_grad(flat, tau=0.01):
    """
    Smooth objective with gradient for L-BFGS-B.
    """
    pts = flat.reshape(-1, 2)
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 1e10, np.zeros_like(flat)
    
    min_a_soft, grad_min_a = _lse_min_area_and_grad(pts, tau)
    
    val = -min_a_soft / ha
    grad = -grad_min_a / ha
    return val, grad.flatten()

def _ring_params_to_points(params):
    """
    Convert ring parameters to point coordinates.
    params: [r1, r2, th1, th2] for 2-ring config (7 outer, 7 inner)
    or [r1, r2, r3, th1, th2, th3] for 3-ring config.
    Here we support variable splits via integer rounding of n1, n2, n3.
    For simplicity and speed, we use a fixed 2-ring (7+7) or 3-ring (5+5+4) structure
    but optimize radii and angles.
    """
    # We will use a flexible 3-ring representation where n1, n2, n3 are fixed in the function signature
    # but the caller decides the split. This function takes the specific split as part of the state or we make it generic.
    # Let's make a generic function that takes a list of (n_i, r_i, th_i)
    pass

def _optimize_ring_params(params, n_split, t0, deadline):
    """
    Optimize ring parameters for a given split of points into rings.
    n_split: list of ints, e.g., [7, 7] or [5, 5, 4].
    params: flat array of [r1, th1, r2, th2, ...]
    """
    n_rings = len(n_split)
    # Each ring has 2 params: radius, angle_offset
    # Total params = 2 * n_rings
    
    def ring_obj(x):
        pts = []
        for i, ni in enumerate(n_split):
            r = x[2*i]
            th = x[2*i + 1]
            if r < 0: r = 0
            angles = th + np.linspace(0, 2*np.pi, ni, endpoint=False)
            ring_pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
            pts.append(ring_pts)
        all_pts = np.vstack(pts)
        # Normalize
        ha = _hull_area(all_pts)
        if ha < 1e-12:
            return 1e10
        all_pts = all_pts / np.sqrt(ha)
        return -_score(all_pts)
    
    try:
        res = scipy.optimize.minimize(ring_obj, params, method='Nelder-Mead', 
                                       options={'maxiter': 2000, 'fatol': 1e-12, 'xatol': 1e-10})
        # Reconstruct points
        pts = []
        for i, ni in enumerate(n_split):
            r = res.x[2*i]
            th = res.x[2*i + 1]
            if r < 0: r = 0
            angles = th + np.linspace(0, 2*np.pi, ni, endpoint=False)
            ring_pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
            pts.append(ring_pts)
        all_pts = np.vstack(pts)
        return _normalize(all_pts)
    except Exception:
        return None

def construct():
    t0 = time.time()
    def left():
        return DEADLINE_FRAC - (time.time() - t0)

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

    top_candidates = [pts for _, pts in scored[:5]]

    def try_opt(x0, method, maxiter, fatol, xatol, use_smooth=False, tau=0.01, use_grad=False):
        nonlocal best_score, best_pts
        if left() <= 4.0:
            return
        try:
            if use_smooth and use_grad:
                obj_func = lambda f: _smooth_neg_score_and_grad(f, tau)
                res = scipy.optimize.minimize(obj_func, x0.flatten(), method=method, jac=True, options={'maxiter': maxiter, 'fatol': fatol, 'xatol': xatol, 'maxfev': 10000, 'maxls': 50})
            elif use_smooth:
                obj_func = lambda f: _smooth_neg_score(f, tau)
                res = scipy.optimize.minimize(obj_func, x0.flatten(), method=method, options={'maxiter': maxiter, 'fatol': fatol, 'xatol': xatol, 'maxfev': 10000, 'maxls': 50})
            else:
                obj_func = _neg_score
                opts = {'maxiter': maxiter, 'fatol': fatol, 'xatol': xatol,
                        'adaptive': True, 'maxfev': 25000}
                res = scipy.optimize.minimize(obj_func, x0.flatten(), method=method, options=opts)
            opt = _normalize(res.x.reshape(-1, 2))
            sc = _score(opt)
            if sc > best_score:
                best_score, best_pts = sc, opt.copy()
        except Exception:
            pass

    # Phase 1: Nelder-Mead on top candidates
    for pts in top_candidates:
        if left() <= 12.0:
            break
        try_opt(pts, 'Nelder-Mead', 15000, 1e-13, 1e-10)

    # Phase 2: L-BFGS-B with smooth surrogate + gradient
    if left() > 25.0:
        for tau in [0.2, 0.1, 0.05, 0.02]:
            if left() <= 8.0:
                break
            try_opt(best_pts, 'L-BFGS-B', 500, 1e-12, 1e-12, use_smooth=True, tau=tau, use_grad=True)
            if left() > 10.0:
                try_opt(best_pts, 'Nelder-Mead', 3000, 1e-13, 1e-10)

    # Phase 3: Structured Ring Optimization (Inspiration 1)
    # Try different splits: [7,7], [6,8], [5,9], [4,10], [5,5,4], [6,4,4], etc.
    if left() > 20.0:
        splits = [
            [7, 7],
            [6, 8],
            [8, 6],
            [5, 9],
            [9, 5],
            [5, 5, 4],
            [4, 5, 5],
            [5, 4, 5],
            [6, 4, 4],
            [4, 6, 4],
            [4, 4, 6],
            [3, 5, 6],
            [6, 5, 3],
        ]
        
        for split in splits:
            if left() <= 5.0:
                break
            # Initial guess: radii spread out, angles 0
            n_rings = len(split)
            init_params = []
            for i in range(n_rings):
                # Radii from 1.0 down to 0.3
                r = 1.0 - i * 0.2
                init_params.extend([max(r, 0.1), 0.0])
            
            opt_pts = _optimize_ring_params(np.array(init_params), split, t0, DEADLINE_FRAC)
            if opt_pts is not None:
                sc = _score(opt_pts)
                if sc > best_score:
                    best_score, best_pts = sc, opt_pts.copy()
                # Also try optimizing the best found so far in this ring structure? 
                # No, just update best.
    
    # Phase 4: Powell on best
    if left() > 15.0:
        try_opt(best_pts, 'Powell', 5000, 1e-13, 1e-11)

    # Phase 5: L-BFGS-B with tight tau
    if left() > 12.0:
        for tau in [0.01, 0.005, 0.002]:
            if left() <= 5.0:
                break
            try_opt(best_pts, 'L-BFGS-B', 400, 1e-14, 1e-14, use_smooth=True, tau=tau, use_grad=True)

    # Phase 6: Final Polish
    if left() > 8.0:
        try_opt(best_pts, 'Nelder-Mead', 10000, 1e-15, 1e-13)
    
    if left() > 4.0:
        try_opt(best_pts, 'Nelder-Mead', 5000, 1e-16, 1e-14)

    return _normalize(best_pts)
