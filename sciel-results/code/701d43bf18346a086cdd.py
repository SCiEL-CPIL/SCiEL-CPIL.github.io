import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14
DEADLINE_FRAC = 145.0  # wall-clock guard (evaluator kills at 150s)

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
_M = len(_TRI)

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

def _three_ring_config(n, r1=0.3, r2=0.6, phase1=0.0, phase2=0.5):
    """Three rings: outer 7, middle 4, inner 3."""
    n_outer, n_mid, n_in = 7, 4, 3
    outer = _regular_polygon(n_outer, 1.0, 0.0)
    mid = _regular_polygon(n_mid, r2, phase2 * 2 * np.pi / n_mid)
    inner = _regular_polygon(n_in, r1, phase1 * 2 * np.pi / n_in)
    return np.vstack([outer, mid, inner])

def _build_structured_starts():
    starts = []
    # Regular polygon baseline (and a few rotations)
    for rot in [0.0, np.pi / N, np.pi / 14, np.pi / 3, np.pi / 7, np.pi / 21]:
        starts.append(_regular_polygon(N, 1.0, rot))
    # Two-ring "near-optimal" structure with varied inner radius and phase
    for frac in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
        for phase in [0.0, 0.5, 0.25, 0.75, 0.1, 0.4]:
            starts.append(_two_ring_config(N, frac, phase))
    # Ellipse configurations with varied aspect ratios and rotations
    for a, b in [(1.0, 0.9), (1.0, 0.85), (1.0, 0.8), (1.0, 0.75), (1.05, 0.95), (1.1, 0.9)]:
        for rot in [0.0, np.pi / 6, np.pi / 4, np.pi / 3]:
            starts.append(_ellipse_config(N, a, b, rot))
    # Three-ring configurations
    for r1 in [0.2, 0.3, 0.4]:
        for r2 in [0.55, 0.65, 0.75]:
            for p1 in [0.0, 0.5]:
                for p2 in [0.0, 0.25, 0.5]:
                    starts.append(_three_ring_config(N, r1, r2, p1, p2))
    return starts

def _lse_min_area_and_grad(points, tau):
    """
    Computes the LSE approximation of min triangle area and its gradient.
    Returns (lse_value, grad_flat)
    """
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    
    v1 = pj - pi
    v2 = pk - pi
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # LSE calculation
    max_a = np.max(areas)
    exp_terms = np.exp(-(areas - max_a) / tau)
    sum_exp = np.sum(exp_terms)
    lse_val = max_a + tau * np.log(sum_exp)
    
    # Gradient calculation
    weights = exp_terms / sum_exp
    
    grad_flat = np.zeros_like(points.flatten())
    
    signs = np.sign(cross)
    if np.any(signs == 0):
        signs[signs == 0] = 1.0
        
    coeff = 0.5 * signs * weights
    
    # Accumulate for p_i
    gi = coeff[:, np.newaxis] * np.column_stack([-v2[:, 1], v2[:, 0]])
    np.add.at(grad_flat, _I * 2 + 0, gi[:, 0])
    np.add.at(grad_flat, _I * 2 + 1, gi[:, 1])
    
    # Accumulate for p_j
    gj = coeff[:, np.newaxis] * np.column_stack([v2[:, 1], -v2[:, 0]])
    np.add.at(grad_flat, _J * 2 + 0, gj[:, 0])
    np.add.at(grad_flat, _J * 2 + 1, gj[:, 1])
    
    # Accumulate for p_k
    gk = coeff[:, np.newaxis] * np.column_stack([-v1[:, 1], v1[:, 0]])
    np.add.at(grad_flat, _K * 2 + 0, gk[:, 0])
    np.add.at(grad_flat, _K * 2 + 1, gk[:, 1])
    
    return lse_val, grad_flat

def _smooth_neg_score(flat, tau):
    pts = flat.reshape(-1, 2)
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 1e10
    lse_val, _ = _lse_min_area_and_grad(pts, tau)
    return -lse_val / ha

def _smooth_neg_score_grad(flat, tau):
    pts = flat.reshape(-1, 2)
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 1e10, np.zeros_like(flat)
    lse_val, grad_lse = _lse_min_area_and_grad(pts, tau)
    grad_f = -grad_lse / ha
    return -lse_val / ha, grad_f

def _annealed_gradient_descent(x0, tau_schedule, maxiter, time_budget):
    """
    Custom gradient descent with annealed LSE.
    Returns the best point found.
    """
    t_start = time.time()
    x = x0.flatten().copy()
    best_x = x.copy()
    best_score = _score(x.reshape(-1, 2))
    
    for step in range(maxiter):
        if time.time() - t_start > time_budget:
            break
        
        # Determine current tau
        tau = tau_schedule[-1][0]
        for t_val, step_thresh in reversed(tau_schedule):
            if step >= step_thresh:
                tau = t_val
                break
        
        f, g = _smooth_neg_score_grad(x, tau)
        if np.isnan(f) or np.isinf(f):
            break
            
        # Simple gradient step with line search (Backtracking)
        alpha = 0.5
        f_new = f
        x_new = x
        for _ in range(30):
            x_trial = x - alpha * g
            f_trial, _ = _smooth_neg_score_grad(x_trial, tau)
            if f_trial < f_new:
                x_new = x_trial
                f_new = f_trial
                break
            alpha *= 0.5
        
        if not np.allclose(x_new, x):
            x = x_new
            sc = _score(x.reshape(-1, 2))
            if sc > best_score:
                best_score = sc
                best_x = x.copy()
        else:
            # If line search fails, try a smaller step or break
            # To avoid getting stuck, we can try a random perturbation
            break
            
    return best_x

def _polish_with_nm(x0, maxiter, fatol, xatol, time_budget):
    """
    Polish a point using Nelder-Mead.
    """
    t_start = time.time()
    def obj(flat):
        if time.time() - t_start > time_budget:
            return 1e10
        return _neg_score(flat)
    
    try:
        res = scipy.optimize.minimize(obj, x0.flatten(), method='Nelder-Mead', 
                                       options={'maxiter': maxiter, 'fatol': fatol, 'xatol': xatol, 'adaptive': True})
        return res.x
    except Exception:
        return x0

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

    # Top candidates for intensive optimization
    top_candidates = [pts for _, pts in scored[:8]]

    # Phase 1: Annealed Gradient Descent on top candidates
    tau_sched = [
        (0.1, 0), (0.05, 200), (0.02, 500), (0.01, 800), (0.005, 1100), (0.002, 1400), (0.001, 1700)
    ]
    
    for i, pts in enumerate(top_candidates):
        if left() <= 15.0:
            break
        time_budget = min(15.0, left() - 5.0)
        x0 = pts.flatten()
        x_res = _annealed_gradient_descent(x0, tau_sched, 2000, time_budget)
        opt = _normalize(x_res.reshape(-1, 2))
        sc = _score(opt)
        if sc > best_score:
            best_score, best_pts = sc, opt.copy()

    # Phase 2: Nelder-Mead polish on the best point
    if left() > 20.0:
        time_budget = min(20.0, left() - 10.0)
        x_res = _polish_with_nm(best_pts.flatten(), 5000, 1e-14, 1e-12, time_budget)
        opt = _normalize(x_res.reshape(-1, 2))
        sc = _score(opt)
        if sc > best_score:
            best_score, best_pts = sc, opt.copy()

    # Phase 3: Final high-precision polish
    if left() > 15.0:
        time_budget = min(15.0, left() - 5.0)
        x_res = _polish_with_nm(best_pts.flatten(), 10000, 1e-16, 1e-14, time_budget)
        opt = _normalize(x_res.reshape(-1, 2))
        sc = _score(opt)
        if sc > best_score:
            best_score, best_pts = sc, opt.copy()

    return _normalize(best_pts)
