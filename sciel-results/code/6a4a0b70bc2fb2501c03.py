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
    """Points on an ellipse."""
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    x = a * np.cos(theta)
    y = b * np.sin(theta)
    pts = np.column_stack([x, y])
    c, s = np.cos(rotation), np.sin(rotation)
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T

def _build_structured_starts():
    starts = []
    # Regular polygon baseline (and a few rotations)
    for rot in [0.0, np.pi / N, np.pi / 14, np.pi / 3]:
        starts.append(_regular_polygon(N, 1.0, rot))
    # Two-ring "near-optimal" structure with varied inner radius and phase
    for frac in [0.45, 0.5, 0.55, 0.6, 0.65]:
        for phase in [0.0, 0.5, 0.25, 0.75]:
            starts.append(_two_ring_config(N, frac, phase))
    # Ellipse configurations with varied aspect ratios and rotations
    for a, b in [(1.0, 0.9), (1.0, 0.85), (1.0, 0.8), (1.0, 0.75), (1.05, 0.95)]:
        for rot in [0.0, np.pi / 6, np.pi / 4]:
            starts.append(_ellipse_config(N, a, b, rot))
    return starts

def _lse_min_area_and_grad(points, tau=0.01):
    """
    Compute LSE approximation of min triangle area AND its gradient.
    
    The LSE min is: min(areas) ≈ max_a - tau * log(sum(exp(-(areas - max_a)/tau)))
    
    For maximizing min area, we want to minimize -min_area, so:
    f = tau * log(sum(exp(-areas/tau)))  (up to constants)
    
    Gradient: df/dp_i = (1/tau) * sum_j w_j * d(areas_j)/dp_i
    where w_j = exp(-areas_j/tau) / sum_k exp(-areas_k/tau)
    
    d(area_j)/dp_i for triangle (i,j,k):
    area = 0.5 * |cross(pj-pi, pk-pi)|
    For signed cross: cross = (pj_x - pi_x)*(pk_y - pi_y) - (pj_y - pi_y)*(pk_x - pi_x)
    d(cross)/d(pi) = (-(pk_y - pi_y), (pk_x - pi_x)) = (-v2_y, v2_x)
    d(cross)/d(pj) = (pk_y - pi_y, -(pk_x - pi_x)) = (v2_y, -v2_x)
    d(cross)/d(pk) = (-(pj_y - pi_y), (pj_x - pi_x)) = (-v1_y, v1_x)
    """
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # LSE min: min(areas) ≈ max_a - tau * log(sum(exp(-(areas - max_a)/tau)))
    max_a = np.max(areas)
    exp_terms = np.exp(-(areas - max_a) / tau)
    sum_exp = np.sum(exp_terms)
    lse_val = max_a - tau * np.log(sum_exp)
    
    # Weights for gradient
    w = exp_terms / sum_exp
    
    # Gradient of LSE min w.r.t. areas:
    # d(lse_min)/d(area_j) = -w_j  (since lse_min = max_a - tau*log(sum_exp))
    # But max_a depends on which area is max, so let's be more careful.
    # Actually, for the soft-min: soft_min(x) = -tau * log(sum(exp(-x/tau)))
    # d(soft_min)/dx_j = exp(-x_j/tau) / sum(exp(-x_k/tau)) = w_j
    # So d(soft_min)/d(area_j) = w_j (positive weight)
    
    # Wait, let me reconsider. We want to maximize min_area.
    # soft_min(areas) = -tau * log(sum(exp(-areas/tau)))
    # d(soft_min)/d(area_j) = exp(-area_j/tau) / sum(exp(-area_k/tau)) = w_j
    # So gradient of soft_min w.r.t. points:
    # d(soft_min)/d(p_i) = sum_j w_j * d(area_j)/d(p_i)
    
    # But we're minimizing -soft_min, so the objective gradient is:
    # d(-soft_min)/d(p_i) = -sum_j w_j * d(area_j)/d(p_i)
    
    # For the full score = min_area / hull_area, we need chain rule.
    # Let's just return the LSE value and let the caller handle the full gradient.
    
    # Compute d(area_j)/d(points) for all triangles
    # area_j = 0.5 * |cross_j|
    # For simplicity, assume all crosses have same sign (convex config)
    # d(cross_j)/d(p_i) for i in triangle (a,b,c):
    #   if i == a: (-v2_y, v2_x)
    #   if i == b: (v2_y, -v2_x)  
    #   if i == c: (-v1_y, v1_x)
    
    grad = np.zeros((N, 2))
    
    # For each triangle, accumulate weighted gradient
    # d(area_j)/d(p_i) = 0.5 * sign(cross_j) * d(cross_j)/d(p_i)
    # For convex configurations, cross should be positive (CCW)
    signs = np.sign(cross)
    # Use 0.5 * sign * d(cross)/dp
    
    # d(cross)/d(p_a) where a=_I: (-v2_y, v2_x)
    g_a_x = -v2[:, 1]
    g_a_y = v2[:, 0]
    
    # d(cross)/d(p_b) where b=_J: (v2_y, -v2_x)
    g_b_x = v2[:, 1]
    g_b_y = -v2[:, 0]
    
    # d(cross)/d(p_c) where c=_K: (-v1_y, v1_x)
    g_c_x = -v1[:, 1]
    g_c_y = v1[:, 0]
    
    # Weighted sums
    w_half = 0.5 * w * signs
    
    # Accumulate into grad
    np.add.at(grad, (_I, 0), w_half * g_a_x)
    np.add.at(grad, (_I, 1), w_half * g_a_y)
    np.add.at(grad, (_J, 0), w_half * g_b_x)
    np.add.at(grad, (_J, 1), w_half * g_b_y)
    np.add.at(grad, (_K, 0), w_half * g_c_x)
    np.add.at(grad, (_K, 1), w_half * g_c_y)
    
    return lse_val, grad

def _smooth_neg_score_and_grad(flat, tau=0.01):
    """
    Smooth objective for L-BFGS-B with analytical gradient.
    Returns (value, gradient) of -min_area_lse / hull_area.
    """
    pts = flat.reshape(-1, 2)
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 1e10, np.zeros(2 * N)
    
    lse_val, lse_grad = _lse_min_area_and_grad(pts, tau)
    
    # Objective: f = -lse_val / ha
    # df/dp = -(lse_grad * ha + lse_val * d(ha)/dp) / ha^2
    # For simplicity, ignore d(ha)/dp (hull area changes slowly)
    # This is an approximation but should work well for local optimization
    
    val = -lse_val / ha
    grad = -lse_grad / ha
    
    return val, grad.flatten()

def _smooth_neg_score_annealed(flat, tau_schedule):
    """
    Smooth objective with annealed tau for L-BFGS-B.
    tau_schedule: list of (tau, iteration_threshold) pairs.
    """
    pts = flat.reshape(-1, 2)
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 1e10
    
    current_iter = _smooth_neg_score_annealed.current_iter
    tau = tau_schedule[-1][0]
    for t, threshold in tau_schedule:
        if current_iter < threshold:
            tau = t
            break
    
    min_a_lse = _lse_min_area_and_grad(pts, tau)[0]
    return -min_a_lse / ha

_smooth_neg_score_annealed.current_iter = 0

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

    def try_opt(x0, method, maxiter, fatol, xatol, use_smooth=False, tau_schedule=None):
        nonlocal best_score, best_pts
        if left() <= 4.0:
            return
        try:
            if use_smooth:
                if tau_schedule is not None:
                    _smooth_neg_score_annealed.current_iter = 0
                    def obj_with_annealing(flat):
                        return _smooth_neg_score_annealed(flat, tau_schedule)
                    obj_func = obj_with_annealing
                else:
                    def obj_func(flat):
                        return _smooth_neg_score_and_grad(flat, tau=0.01)[0]
            else:
                obj_func = _neg_score
            
            opts = {'maxiter': maxiter, 'fatol': fatol, 'xatol': xatol,
                    'adaptive': True, 'maxfev': 25000}
            if method == 'L-BFGS-B':
                opts['maxfev'] = 10000
                opts['maxls'] = 50
            if method == 'Nelder-Mead':
                opts['adaptive'] = True
            
            if use_smooth and tau_schedule is not None:
                def callback(flat):
                    _smooth_neg_score_annealed.current_iter += 1
                    return False
                opts['callback'] = callback
            
            res = scipy.optimize.minimize(obj_func, x0.flatten(), method=method, options=opts)
            opt = _normalize(res.x.reshape(-1, 2))
            sc = _score(opt)
            if sc > best_score:
                best_score, best_pts = sc, opt.copy()
        except Exception:
            pass

    def try_opt_with_grad(x0, method, maxiter, fatol, xatol, tau):
        """L-BFGS-B with analytical gradient."""
        nonlocal best_score, best_pts
        if left() <= 4.0:
            return
        try:
            def obj_and_grad(flat):
                val, grad = _smooth_neg_score_and_grad(flat, tau)
                return val, grad
            
            opts = {'maxiter': maxiter, 'fatol': fatol, 'xatol': xatol,
                    'maxfev': 10000, 'maxls': 50}
            
            res = scipy.optimize.minimize(obj_and_grad, x0.flatten(), method=method,
                                         jac=True, options=opts)
            opt = _normalize(res.x.reshape(-1, 2))
            sc = _score(opt)
            if sc > best_score:
                best_score, best_pts = sc, opt.copy()
        except Exception:
            pass

    # Intensive optimization on top candidates
    for pts in top_candidates:
        if left() <= 10.0:
            break
        
        # 1. Nelder-Mead (fast, robust)
        try_opt(pts, 'Nelder-Mead', 20000, 1e-13, 1e-10)
        
        if left() <= 15.0:
            break
            
        # 2. Powell (slower, better local search)
        try_opt(best_pts, 'Powell', 6000, 1e-13, 1e-11)
        
        if left() <= 20.0:
            break

        # 3. L-BFGS-B with analytical gradient (much faster than finite diff)
        try_opt_with_grad(best_pts, 'L-BFGS-B', 5000, 1e-12, 1e-12, tau=0.01)
        
        if left() <= 25.0:
            break

        # 4. Final Polish: Tight Nelder-Mead
        try_opt(best_pts, 'Nelder-Mead', 5000, 1e-15, 1e-12)
        
        if left() <= 30.0:
            break

        # 5. Additional polish: Very tight Powell
        try_opt(best_pts, 'Powell', 3000, 1e-14, 1e-13)

    # Final high-precision polish if time allows
    if left() > 15.0:
        # Use smaller tau for more precise min tracking
        try_opt_with_grad(best_pts, 'L-BFGS-B', 3000, 1e-14, 1e-14, tau=0.005)
    
    if left() > 10.0:
        try_opt(best_pts, 'Nelder-Mead', 10000, 1e-16, 1e-14)
    
    return _normalize(best_pts)
