import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14
DEADLINE = 142.0

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
        scaled = points / np.sqrt(ha)
        ha_check = _hull_area(scaled)
        if ha_check < 1e-6:
            return points
        return scaled
    except Exception:
        return points

def _regular_polygon(n, radius=1.0, rotation=0.0):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    return np.column_stack([radius * np.cos(a), radius * np.sin(a)])

def _two_heptagons(r_inner, phi):
    outer = _regular_polygon(7, 1.0, 0.0)
    inner = _regular_polygon(7, r_inner, phi)
    return np.vstack([outer, inner])

def _score_two_heptagons(params):
    try:
        r_inner, phi = params
        if r_inner < 0:
            r_inner = 0.0
        points = _two_heptagons(r_inner, phi)
        return _score(points)
    except Exception:
        return 0.0

def _neg_score_two_heptagons(params):
    return -_score_two_heptagons(params)

def _exact_maxmin_obj_grad(points):
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)

    ha = _hull_area(points)
    if ha < 1e-12:
        return 1e10, np.zeros((N, 2))

    m = areas.min()
    tol = 1e-10 * max(1.0, m)
    mask = areas <= (m + tol)
    cnt = mask.sum()
    w = np.zeros(_NTRI)
    w[mask] = 1.0 / cnt

    g_a_x = v1[:, 1] - v2[:, 1]
    g_a_y = v2[:, 0] - v1[:, 0]
    g_b_x = v2[:, 1]
    g_b_y = -v2[:, 0]
    g_c_x = -v1[:, 1]
    g_c_y = v1[:, 0]

    wh = 0.5 * w
    grad = np.zeros((N, 2))
    np.add.at(grad, (_I, 0), wh * g_a_x)
    np.add.at(grad, (_I, 1), wh * g_a_y)
    np.add.at(grad, (_J, 0), wh * g_b_x)
    np.add.at(grad, (_J, 1), wh * g_b_y)
    np.add.at(grad, (_K, 0), wh * g_c_x)
    np.add.at(grad, (_K, 1), wh * g_c_y)

    val = -m / ha
    grad = -grad / ha
    return val, grad

def _soft_min_score(x, tau):
    points = x.reshape(-1, 2)
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    areas = np.maximum(areas, 1e-12)
    
    scaled = -areas / tau
    max_scaled = np.max(scaled)
    sum_exp = np.sum(np.exp(scaled - max_scaled))
    soft_min_val = -tau * (max_scaled + np.log(sum_exp))
    
    return soft_min_val / ha

def _hull_area_grad(points):
    try:
        hull = scipy.spatial.ConvexHull(points)
        verts = hull.vertices
        n_v = len(verts)
        if n_v < 3:
            return np.zeros((N, 2))
        centroid = points[verts].mean(axis=0)
        angles = np.arctan2(points[verts, 1] - centroid[1], points[verts, 0] - centroid[0])
        order = np.argsort(angles)
        ordered_verts = verts[order]
        
        grad = np.zeros((N, 2))
        for k in range(n_v):
            vi = ordered_verts[k]
            vp = ordered_verts[(k - 1) % n_v]
            vn = ordered_verts[(k + 1) % n_v]
            grad[vi, 0] += 0.5 * (points[vn, 1] - points[vp, 1])
            grad[vi, 1] += 0.5 * (points[vp, 0] - points[vn, 0])
        return grad
    except Exception:
        return np.zeros((N, 2))

def _soft_min_score_grad(x, tau):
    points = x.reshape(-1, 2)
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0, np.zeros((N, 2))
    except Exception:
        return 0.0, np.zeros((N, 2))
    
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    areas = np.maximum(areas, 1e-12)
    
    scaled = -areas / tau
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    sum_exp = np.sum(exp_scaled)
    probs = exp_scaled / sum_exp
    
    soft_min_val = -tau * (max_scaled + np.log(sum_exp))
    
    # Gradient of soft_min w.r.t areas is -probs
    # We want to maximize soft_min, so we minimize -soft_min.
    # d(-soft_min)/d(areas) = probs
    # But we are minimizing -score = - (min_area / ha)
    # Let's stick to the convention: return (f, grad) where f is to be minimized.
    # f = - soft_min_val / ha
    # d(soft_min)/d(area_k) = -prob_k
    # d(f)/d(area_k) = - (1/ha) * d(soft_min)/d(area_k) = prob_k / ha
    
    g_areas = probs
    
    g_a_x = v1[:, 1] - v2[:, 1]
    g_a_y = v2[:, 0] - v1[:, 0]
    g_b_x = v2[:, 1]
    g_b_y = -v2[:, 0]
    g_c_x = -v1[:, 1]
    g_c_y = v1[:, 0]
    
    grad_num = np.zeros((N, 2))
    np.add.at(grad_num, (_I, 0), g_areas * g_a_x)
    np.add.at(grad_num, (_I, 1), g_areas * g_a_y)
    np.add.at(grad_num, (_J, 0), g_areas * g_b_x)
    np.add.at(grad_num, (_J, 1), g_areas * g_b_y)
    np.add.at(grad_num, (_K, 0), g_areas * g_c_x)
    np.add.at(grad_num, (_K, 1), g_areas * g_c_y)
    
    grad_ha = _hull_area_grad(points)
    
    # f = - soft_min_val / ha
    # df = - (d(soft_min_val) / ha - soft_min_val * d(ha) / ha^2)
    # d(soft_min_val) = sum (d(soft_min)/d(area_k) * grad_area_k . dx)
    # d(soft_min)/d(area_k) = -prob_k
    # So d(soft_min_val) = - sum (prob_k * grad_area_k . dx) = - (grad_num . dx)  [if grad_num uses -probs]
    # Wait, let's re-evaluate carefully.
    
    # soft_min = -tau * logsumexp(-areas/tau)
    # d(soft_min)/d(area_k) = -tau * (1/sum_exp) * exp(-area_k/tau) * (-1/tau) = exp(-area_k/tau)/sum_exp = prob_k
    # So d(soft_min)/d(area_k) = prob_k.
    
    # f = - soft_min / ha
    # df = - (1/ha) * d(soft_min) + (soft_min / ha^2) * d(ha)
    # d(soft_min) = sum (prob_k * d(area_k))
    # d(area_k) = 0.5 * (v1 x v2) derivative.
    # grad_area_k w.r.t points is calculated in g_a, g_b, g_c.
    # So d(soft_min) = sum_k prob_k * (grad_area_k . dx)
    # Let grad_soft = sum_k prob_k * grad_area_k.
    # Then df = - (1/ha) * grad_soft . dx + (soft_min / ha^2) * grad_ha . dx
    
    # In my code above, I calculated grad_num using g_areas = probs.
    # So grad_num = sum_k prob_k * grad_area_k = grad_soft.
    
    # f = - soft_min / ha
    # grad_f = - grad_soft / ha + soft_min * grad_ha / ha^2
    
    val = -soft_min_val / ha
    grad = -grad_num / ha + soft_min_val * grad_ha / (ha * ha)
    
    return val, grad

def _ring_configuration(params):
    r1, r2, r3, r4, th1, th2, th3, th4, n1, n2, n3, n4 = params
    n1 = int(round(n1))
    n2 = int(round(n2))
    n3 = int(round(n3))
    n4 = int(round(n4))
    
    total = n1 + n2 + n3 + n4
    while total > N:
        if n4 >= 1:
            n4 -= 1
        elif n3 >= 1:
            n3 -= 1
        elif n2 >= 1:
            n2 -= 1
        else:
            n1 -= 1
        total -= 1
    while total < N:
        n4 += 1
        total += 1
    
    points = []
    
    if n1 > 0:
        angles = th1 + np.linspace(0, 2 * np.pi, n1, endpoint=False)
        points.append(np.column_stack((r1 * np.cos(angles), r1 * np.sin(angles))))
    
    if n2 > 0:
        angles = th2 + np.linspace(0, 2 * np.pi, n2, endpoint=False)
        points.append(np.column_stack((r2 * np.cos(angles), r2 * np.sin(angles))))
    
    if n3 > 0:
        angles = th3 + np.linspace(0, 2 * np.pi, n3, endpoint=False)
        points.append(np.column_stack((r3 * np.cos(angles), r3 * np.sin(angles))))
    
    if n4 > 0:
        angles = th4 + np.linspace(0, 2 * np.pi, n4, endpoint=False)
        points.append(np.column_stack((r4 * np.cos(angles), r4 * np.sin(angles))))
    
    if not points:
        return np.zeros((N, 2))
    
    result = np.vstack(points)
    
    if len(result) < N:
        extra = np.zeros((N - len(result), 2))
        result = np.vstack([result, extra])
    elif len(result) > N:
        result = result[:N]
    
    return result

def _neg_ring_score(params):
    try:
        points = _ring_configuration(params)
        return -_score(points)
    except Exception:
        return 0.0

def _convex_polygon_from_edge_lengths(edge_lengths, angles=None):
    n = len(edge_lengths)
    if angles is None:
        angles = np.full(n, 2 * np.pi / n)
    
    angle_sum = np.sum(angles)
    if angle_sum < 1e-10:
        angles = np.full(n, 2 * np.pi / n)
    else:
        angles = angles / angle_sum * 2 * np.pi
    
    directions = np.cumsum(angles)
    directions = np.roll(directions, -1) - angles / 2
    
    edge_vectors = np.column_stack([
        edge_lengths * np.cos(directions),
        edge_lengths * np.sin(directions)
    ])
    
    vertices = np.cumsum(edge_vectors, axis=0)
    vertices -= vertices.mean(axis=0)
    
    return vertices

def _neg_convex_poly_score(params):
    try:
        n = N
        e = params[:n]
        a = params[n:n + n - 1]
        
        e = np.maximum(e, 0.01)
        a = np.maximum(a, 0.01)
        
        a_full = np.append(a, 2 * np.pi - np.sum(a))
        a_full = np.maximum(a_full, 0.01)
        
        a_sum = np.sum(a_full)
        a_full = a_full / a_sum * 2 * np.pi
        
        vertices = _convex_polygon_from_edge_lengths(e, a_full)
        
        try:
            hull = scipy.spatial.ConvexHull(vertices)
            if len(hull.vertices) < n:
                return 1e10
        except Exception:
            return 1e10
        
        return -_score(vertices)
    except Exception:
        return 1e10

def _convex_poly_from_edge_and_turn(params):
    """
    Parameterization: edge_lengths (N), turn_angles (N-1).
    The last turn angle is determined by closure: sum(turns) = 2*pi.
    We parameterize turns as positive values, normalize to sum to 2*pi.
    This guarantees a convex polygon that closes.
    """
    n = N
    e = params[:n]
    t = params[n:]
    
    e = np.maximum(e, 0.01)
    t = np.maximum(t, 0.01)
    
    # Normalize turn angles to sum to 2*pi
    t_full = np.append(t, 2 * np.pi - np.sum(t))
    t_full = np.maximum(t_full, 0.01)
    t_sum = np.sum(t_full)
    t_full = t_full / t_sum * 2 * np.pi
    
    # Build polygon
    vertices = np.zeros((n, 2))
    vertices[0] = [0, 0]
    current_angle = 0.0
    
    for i in range(n):
        vertices[i+1] = vertices[i] + e[i] * np.array([np.cos(current_angle), np.sin(current_angle)])
        current_angle += t_full[i]
    
    vertices = vertices[:n]
    vertices -= vertices.mean(axis=0)
    
    return vertices

def _neg_convex_poly_score2(params):
    """
    Better convex polygon parameterization that guarantees closure and convexity.
    """
    try:
        n = N
        e = params[:n]
        t = params[n:]
        
        e = np.maximum(e, 0.01)
        t = np.maximum(t, 0.01)
        
        # Normalize turn angles to sum to 2*pi
        t_full = np.append(t, 2 * np.pi - np.sum(t))
        t_full = np.maximum(t_full, 0.01)
        t_sum = np.sum(t_full)
        t_full = t_full / t_sum * 2 * np.pi
        
        # Build polygon
        vertices = np.zeros((n, 2))
        vertices[0] = [0, 0]
        current_angle = 0.0
        
        for i in range(n):
            vertices[i + 1] = vertices[i] + e[i] * np.array([np.cos(current_angle), np.sin(current_angle)])
            current_angle += t_full[i]
        
        vertices = vertices[:n]
        vertices -= vertices.mean(axis=0)
        
        # Check convexity
        try:
            hull = scipy.spatial.ConvexHull(vertices)
            if len(hull.vertices) < n:
                return 1e10
        except Exception:
            return 1e10
        
        return -_score(vertices)
    except Exception:
        return 1e10

def construct():
    t0 = time.time()
    
    def left():
        return DEADLINE - (time.time() - t0)
    
    best_score = 0.0
    best_pts = _normalize(_regular_polygon(N, 1.0, 0.0))
    
    def try_update(points):
        nonlocal best_score, best_pts
        sc = _score(points)
        if sc > best_score:
            best_score = sc
            best_pts = points.copy()
        return sc

    # Phase 0: Two-nested-heptagons 2D global search
    if left() > 5.0:
        r_vals = np.linspace(0.0, 1.0, 50)
        phi_vals = np.linspace(0.0, np.pi / 7, 50, endpoint=False)
        
        for r in r_vals:
            if left() <= 3.0:
                break
            for phi in phi_vals:
                pts = _two_heptagons(r, phi)
                try_update(pts)
        
        if left() > 5.0:
            best_r, best_phi, best_s = 0.0, 0.0, 0.0
            r_vals = np.linspace(0.0, 1.0, 30)
            phi_vals = np.linspace(0.0, np.pi / 7, 30, endpoint=False)
            for r in r_vals:
                for phi in phi_vals:
                    s = _score_two_heptagons([r, phi])
                    if s > best_s:
                        best_s, best_r, best_phi = s, r, phi
            
            gr = 0.6180339887498949
            a_r, b_r = max(0.0, best_r - 0.1), min(1.0, best_r + 0.1)
            c_r = b_r - gr * (b_r - a_r)
            d_r = a_r + gr * (b_r - a_r)
            fc = _score_two_heptagons([c_r, best_phi])
            fd = _score_two_heptagons([d_r, best_phi])
            for _ in range(40):
                if left() <= 2.0:
                    break
                if fc < fd:
                    a_r = c_r
                    c_r = d_r
                    fc = fd
                    d_r = b_r - gr * (b_r - a_r)
                    fd = _score_two_heptagons([d_r, best_phi])
                else:
                    b_r = d_r
                    d_r = c_r
                    fd = fc
                    c_r = a_r + gr * (b_r - a_r)
                    fc = _score_two_heptagons([c_r, best_phi])
            best_r = (a_r + b_r) / 2
            
            a_p, b_p = 0.0, np.pi / 7
            c_p = b_p - gr * (b_p - a_p)
            d_p = a_p + gr * (b_p - a_p)
            fc = _score_two_heptagons([best_r, c_p])
            fd = _score_two_heptagons([best_r, d_p])
            for _ in range(40):
                if left() <= 2.0:
                    break
                if fc < fd:
                    a_p = c_p
                    c_p = d_p
                    fc = fd
                    d_p = b_p - gr * (b_p - a_p)
                    fd = _score_two_heptagons([best_r, d_p])
                else:
                    b_p = d_p
                    d_p = c_p
                    fd = fc
                    c_p = a_p + gr * (b_p - a_p)
                    fc = _score_two_heptagons([best_r, c_p])
            best_phi = (a_p + b_p) / 2
            
            pts = _two_heptagons(best_r, best_phi)
            try_update(pts)

    # Phase 1: Convex Polygon Parameterization with guaranteed closure
    if left() > 35.0:
        # Use the new parameterization that guarantees a closed convex polygon
        n = N
        init_e = np.ones(n)
        init_t = np.ones(n - 1) * (2 * np.pi / n)
        x0 = np.concatenate([init_e, init_t])
        
        bounds_poly = [(0.5, 1.5) for _ in range(n)] + [(0.3, 2 * np.pi / 3) for _ in range(n - 1)]
        
        try:
            res = scipy.optimize.differential_evolution(
                _neg_convex_poly_score2,
                bounds_poly,
                seed=123,
                maxiter=60,
                tol=1e-5,
                popsize=12,
                mutation=(0.5, 1.0),
                recombination=0.8,
                polish=True
            )
            if res.fun < 0:
                e = res.x[:n]
                t = res.x[n:]
                
                t_full = np.append(t, 2 * np.pi - np.sum(t))
                t_full = np.maximum(t_full, 0.01)
                t_sum = np.sum(t_full)
                t_full = t_full / t_sum * 2 * np.pi
                
                vertices = np.zeros((n, 2))
                vertices[0] = [0, 0]
                current_angle = 0.0
                for i in range(n):
                    vertices[i + 1] = vertices[i] + e[i] * np.array([np.cos(current_angle), np.sin(current_angle)])
                    current_angle += t_full[i]
                vertices = vertices[:n]
                vertices -= vertices.mean(axis=0)
                
                try_update(vertices)
        except Exception:
            pass

    # Phase 2: Differential evolution on 4-ring parameters (global search)
    if left() > 30.0:
        bounds = [
            (0.3, 1.5),
            (0.1, 1.0),
            (0.0, 0.7),
            (0.0, 0.4),
            (0, 2 * np.pi),
            (0, 2 * np.pi),
            (0, 2 * np.pi),
            (0, 2 * np.pi),
            (1, 14),
            (1, 14),
            (1, 14),
            (1, 14),
        ]
        
        rng = np.random.default_rng(42)
        for restart in range(2):
            if left() <= 20.0:
                break
            try:
                res = scipy.optimize.differential_evolution(
                    _neg_ring_score,
                    bounds,
                    seed=rng.integers(0, 2**31),
                    maxiter=40,
                    tol=1e-5,
                    popsize=8,
                    mutation=(0.3, 1.2),
                    recombination=0.9,
                    polish=True,
                    init='latinhypercube'
                )
                if res.success or res.fun < 0:
                    points = _ring_configuration(res.x)
                    try_update(points)
            except Exception:
                pass

    # Phase 3: Structured starts + gradient-based local refinement
    if left() > 20.0:
        starts = [best_pts.copy()]
        starts.append(_normalize(_regular_polygon(N, 1.0, 0.0)))
        
        a_out = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        a_in = a_out + np.pi / 7
        for r_in in [0.4, 0.6, 0.8]:
            starts.append(_normalize(np.vstack((
                np.column_stack((np.cos(a_out), np.sin(a_out))),
                np.column_stack((r_in * np.cos(a_in), r_in * np.sin(a_in)))
            ))))
        
        a1 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        a2 = a1 + np.pi / 5
        a3 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
        for (r2, r3) in [(0.5, 0.2), (0.6, 0.35)]:
            starts.append(_normalize(np.vstack((
                np.column_stack((np.cos(a1), np.sin(a1))),
                np.column_stack((r2 * np.cos(a2), r2 * np.sin(a2))),
                np.column_stack((r3 * np.cos(a3), r3 * np.sin(a3)))
            ))))
        
        rng2 = np.random.default_rng(123)
        for _ in range(4):
            if left() <= 10.0:
                break
            jittered = best_pts + rng2.normal(0, 0.05, (N, 2))
            starts.append(_normalize(jittered))
        
        def try_opt(x0, method, maxiter, fatol, xatol, use_grad=False):
            nonlocal best_score, best_pts
            if left() <= 3.0:
                return
            try:
                if use_grad:
                    def obj_and_grad(flat):
                        val, grad = _exact_maxmin_obj_grad(flat.reshape(-1, 2))
                        return val, grad.flatten()
                    opts = {'maxiter': maxiter, 'fatol': fatol, 'xatol': xatol,
                            'maxfev': 8000, 'maxls': 30}
                    res = scipy.optimize.minimize(obj_and_grad, x0.flatten(), method=method,
                                                  jac=True, options=opts)
                else:
                    opts = {'maxiter': maxiter, 'fatol': fatol, 'xatol': xatol, 'maxfev': 12000}
                    if method == 'Nelder-Mead':
                        opts['adaptive'] = True
                    res = scipy.optimize.minimize(_neg_score, x0.flatten(), method=method, options=opts)
                opt = _normalize(res.x.reshape(-1, 2))
                sc = _score(opt)
                if sc > best_score:
                    best_score, best_pts = sc, opt.copy()
            except Exception:
                pass
        
        for start in starts:
            if left() <= 8.0:
                break
            try_opt(start, 'L-BFGS-B', 1500, 1e-12, 1e-11, use_grad=True)
            if left() <= 6.0:
                break
            try_opt(start, 'Nelder-Mead', 2000, 1e-12, 1e-9)

    # Phase 4: Soft-min annealing
    if left() > 25.0:
        soft_starts = [best_pts.copy()]
        rng_soft = np.random.default_rng(777)
        # Add more diverse starts
        for jitter in [0.05, 0.1, 0.2]:
            if left() <= 12.0:
                break
            jittered = best_pts + rng_soft.normal(0, jitter, (N, 2))
            soft_starts.append(_normalize(jittered))
        
        a_out = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        a_in = a_out + np.pi / 7
        for r_in in [0.3, 0.5, 0.7]:
            if left() <= 10.0:
                break
            ring_pts = np.vstack((
                np.column_stack((np.cos(a_out), np.sin(a_out))),
                np.column_stack((r_in * np.cos(a_in), r_in * np.sin(a_in)))
            ))
            soft_starts.append(_normalize(ring_pts))
        
        tau_vals = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
        for tau in tau_vals:
            if left() <= 6.0:
                break
            best_in_tau = None
            best_sc_in_tau = best_score
            for start in soft_starts[:]:
                if left() <= 4.0:
                    break
                try:
                    def obj_and_grad(flat, _tau=tau):
                        val, grad = _soft_min_score_grad(flat, _tau)
                        return val, grad.flatten()
                    res = scipy.optimize.minimize(obj_and_grad, start.flatten(), method='L-BFGS-B',
                                                  jac=True, options={'maxiter': 1000, 'fatol': 1e-14, 'xatol': 1e-13, 'maxfev': 8000})
                    opt = _normalize(res.x.reshape(-1, 2))
                    sc = _score(opt)
                    if sc > best_sc_in_tau:
                        best_sc_in_tau = sc
                        best_in_tau = opt.copy()
                except Exception:
                    pass
            if best_in_tau is not None:
                best_score = best_sc_in_tau
                best_pts = best_in_tau
                soft_starts = [best_in_tau.copy()]
                if len(soft_starts) < 3 and left() > 8.0:
                    soft_starts.append(best_pts + rng_soft.normal(0, 0.05, (N, 2)))

    # Phase 5: Simulated Annealing for global escape
    if left() > 20.0:
        current_pts = best_pts.copy()
        current_score = best_score
        T = 0.01
        T_min = 1e-5
        cooling = 0.95
        steps = 200
        rng_sa = np.random.default_rng(999)
        
        for step in range(steps):
            if left() <= 10.0:
                break
            if T < T_min:
                break
                
            # Generate neighbor
            noise = rng_sa.normal(0, T, (N, 2))
            new_pts = current_pts + noise
            new_pts = _normalize(new_pts)
            new_score = _score(new_pts)
            
            delta = new_score - current_score
            if delta > 0 or rng_sa.random() < np.exp(delta / T):
                current_pts = new_pts
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_pts = current_pts.copy()
            
            T *= cooling

    # Phase 6: Fine polish
    if left() > 15.0:
        try_opt(best_pts, 'L-BFGS-B', 4000, 1e-13, 1e-12, use_grad=True)
    if left() > 12.0:
        try_opt(best_pts, 'Nelder-Mead', 5000, 1e-14, 1e-12)
    if left() > 10.0:
        try_opt(best_pts, 'L-BFGS-B', 3000, 1e-14, 1e-13, use_grad=True)
    if left() > 8.0:
        try_opt(best_pts, 'Nelder-Mead', 4000, 1e-15, 1e-14)
    if left() > 6.0:
        try_opt(best_pts, 'L-BFGS-B', 2500, 1e-15, 1e-14, use_grad=True)
    if left() > 4.5:
        try_opt(best_pts, 'Nelder-Mead', 3000, 1e-16, 1e-15)
    if left() > 3.0:
        try_opt(best_pts, 'L-BFGS-B', 2000, 1e-16, 1e-15, use_grad=True)
    if left() > 2.0:
        try_opt(best_pts, 'Nelder-Mead', 2000, 1e-17, 1e-16)
    
    return _normalize(best_pts)
