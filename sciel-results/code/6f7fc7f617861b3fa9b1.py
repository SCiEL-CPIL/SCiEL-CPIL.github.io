import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14
DEADLINE = 142.0  # wall-clock guard (8s safety margin before 150s kill)

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

def _two_heptagons(r_inner, phi):
    """Two nested regular heptagons: outer at radius 1, inner at r_inner rotated by phi."""
    outer = _regular_polygon(7, 1.0, 0.0)
    inner = _regular_polygon(7, r_inner, phi)
    return np.vstack([outer, inner])

def _score_two_heptagons(params):
    """Score for two-nested-heptagons parameterization."""
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
    """
    Returns (value, grad) for f(points) = -min_triangle_area(points) / hull_area(points).
    Uses the EXACT minimum (no LSE). The subgradient of min_j a_j is a unit vector on
    the set of minimizers, which is valid for maximizing the minimum.
    """
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

def _soft_min_score_grad(x, tau):
    """Returns (value, grad) for soft-min objective."""
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
    probs = exp_scaled / sum_exp  # softmax weights
    
    soft_min_val = -tau * (max_scaled + np.log(sum_exp))
    
    # Gradient of soft_min w.r.t. areas
    g_areas = -probs
    
    # Gradient of areas w.r.t. points
    g_a_x = v1[:, 1] - v2[:, 1]
    g_a_y = v2[:, 0] - v1[:, 0]
    g_b_x = v2[:, 1]
    g_b_y = -v2[:, 0]
    g_c_x = -v1[:, 1]
    g_c_y = v1[:, 0]
    
    grad = np.zeros((N, 2))
    np.add.at(grad, (_I, 0), g_areas * g_a_x)
    np.add.at(grad, (_I, 1), g_areas * g_a_y)
    np.add.at(grad, (_J, 0), g_areas * g_b_x)
    np.add.at(grad, (_J, 1), g_areas * g_b_y)
    np.add.at(grad, (_K, 0), g_areas * g_c_x)
    np.add.at(grad, (_K, 1), g_areas * g_c_y)
    
    val = -soft_min_val / ha
    grad = -grad / ha
    return val, grad

def _ring_configuration(params):
    """
    Generate points from 4-ring parameters.
    params: [r1, r2, r3, r4, th1, th2, th3, th4, n1, n2, n3, n4]
    where n1+n2+n3+n4=14, r_i >= 0, theta offsets in [0, 2pi)
    """
    r1, r2, r3, r4, th1, th2, th3, th4, n1, n2, n3, n4 = params
    n1 = int(round(n1))
    n2 = int(round(n2))
    n3 = int(round(n3))
    n4 = int(round(n4))
    
    # Ensure we have exactly 14 points
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
            best_pts = _normalize(points).copy()
    
    # Phase 0: Quick 2D optimization of two-nested-heptagons
    if left() > 10.0:
        try:
            # Grid search for initial guess
            best_2d_score = 0.0
            best_2d_params = [0.5, 0.0]
            for r in np.linspace(0.1, 0.9, 15):
                for phi in np.linspace(0, 2*np.pi, 15, endpoint=False):
                    sc = _score_two_heptagons([r, phi])
                    if sc > best_2d_score:
                        best_2d_score = sc
                        best_2d_params = [r, phi]
            
            # Nelder-Mead refinement in 2D
            res = scipy.optimize.minimize(
                _neg_score_two_heptagons, best_2d_params, method='Nelder-Mead',
                options={'maxiter': 5000, 'fatol': 1e-15, 'xatol': 1e-12}
            )
            pts = _two_heptagons(res.x[0], res.x[1])
            try_update(pts)
        except Exception:
            pass
    
    # Phase 1: 4-ring parameterization with differential evolution
    if left() > 30.0:
        try:
            # Define bounds for ring parameters
            # r1, r2, r3, r4: radii (0 to 1.2)
            # th1, th2, th3, th4: angles (0 to 2pi)
            # n1, n2, n3, n4: counts (1 to 7, sum=14)
            bounds = [
                (0.0, 1.2), (0.0, 1.2), (0.0, 1.2), (0.0, 1.2),  # radii
                (0.0, 2*np.pi), (0.0, 2*np.pi), (0.0, 2*np.pi), (0.0, 2*np.pi),  # angles
                (1.0, 7.0), (1.0, 7.0), (1.0, 7.0), (1.0, 7.0)  # counts
            ]
            
            # Use differential evolution with limited population
            de_res = scipy.optimize.differential_evolution(
                _neg_ring_score, bounds,
                maxiter=200, popsize=10, seed=42,
                tol=1e-10, mutation=(0.5, 1.5), recombination=0.8
            )
            if de_res.success or de_res.fun < 0:
                pts = _ring_configuration(de_res.x)
                try_update(pts)
        except Exception:
            pass
    
    # Phase 2: Soft-min annealing from current best
    if left() > 20.0:
        try:
            x = best_pts.flatten().copy()
            # Anneal tau from large to small
            for tau in [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
                if left() <= 5.0:
                    break
                def obj_grad(f, t=tau):
                    return _soft_min_score_grad(f, t)
                res = scipy.optimize.minimize(
                    obj_grad, x, jac=True, method='L-BFGS-B',
                    options={'maxiter': 200, 'gtol': 1e-10, 'ftol': 1e-14}
                )
                x = res.x
                try_update(x.reshape(-1, 2))
        except Exception:
            pass
    
    # Phase 3: Exact subgradient polish
    if left() > 10.0:
        try:
            x = best_pts.flatten().copy()
            def obj_grad(f):
                val, g = _exact_maxmin_obj_grad(f.reshape(-1, 2))
                return val, g.flatten()
            res = scipy.optimize.minimize(
                obj_grad, x, jac=True, method='L-BFGS-B',
                options={'maxiter': 100, 'gtol': 1e-12, 'ftol': 1e-15}
            )
            try_update(res.x.reshape(-1, 2))
        except Exception:
            pass
    
    # Phase 4: Final Nelder-Mead polish
    if left() > 5.0:
        try:
            res = scipy.optimize.minimize(
                _neg_score, best_pts.flatten(), method='Nelder-Mead',
                options={'maxiter': 5000, 'fatol': 1e-15, 'xatol': 1e-13, 'adaptive': True}
            )
            try_update(res.x.reshape(-1, 2))
        except Exception:
            pass
    
    # Phase 5: One more round of soft-min with very small tau
    if left() > 5.0:
        try:
            x = best_pts.flatten().copy()
            for tau in [0.0005, 0.0002, 0.0001]:
                if left() <= 2.0:
                    break
                def obj_grad(f, t=tau):
                    return _soft_min_score_grad(f, t)
                res = scipy.optimize.minimize(
                    obj_grad, x, jac=True, method='L-BFGS-B',
                    options={'maxiter': 100, 'gtol': 1e-12, 'ftol': 1e-15}
                )
                x = res.x
                try_update(x.reshape(-1, 2))
        except Exception:
            pass
    
    # Final exact polish
    if left() > 3.0:
        try:
            x = best_pts.flatten().copy()
            def obj_grad(f):
                val, g = _exact_maxmin_obj_grad(f.reshape(-1, 2))
                return val, g.flatten()
            res = scipy.optimize.minimize(
                obj_grad, x, jac=True, method='L-BFGS-B',
                options={'maxiter': 50, 'gtol': 1e-13, 'ftol': 1e-15}
            )
            try_update(res.x.reshape(-1, 2))
        except Exception:
            pass
    
    return _normalize(best_pts)
