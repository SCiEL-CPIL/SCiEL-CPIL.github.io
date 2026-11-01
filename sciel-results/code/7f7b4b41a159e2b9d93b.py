import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14

# Precompute triangle indices once
_IDX = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])
_I, _J, _K = _IDX[:, 0], _IDX[:, 1], _IDX[:, 2]

def compute_score(points):
    """Compute min triangle area / convex hull area, vectorized."""
    n = len(points)
    if n < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0

    p_i = points[_I]
    p_j = points[_J]
    p_k = points[_K]

    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)

    min_area = np.min(areas)
    if min_area < 1e-12:
        return 0.0
    return min_area / hull_area

def _smooth_min_score(x, tau):
    """
    Smoothed negative objective:
    -soft_min(areas) / hull_area, where soft_min approximates min(areas) as tau -> 0.
    """
    points = x.reshape((N, 2))
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0

    p_i = points[_I]
    p_j = points[_J]
    p_k = points[_K]

    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    areas = np.maximum(areas, 1e-12)

    scaled = -areas / tau
    max_scaled = np.max(scaled)
    sum_exp = np.sum(np.exp(scaled - max_scaled))
    soft_min_val = -tau * (max_scaled + np.log(sum_exp))
    return -soft_min_val / hull_area

def neg_score(x):
    points = x.reshape((N, 2))
    return -compute_score(points)

def _ring_configuration(params):
    """
    Build 14 points from up to 4 concentric rings.
    params layout:
      [r1, r2, r3, r4, th1, th2, th3, th4, n1, n2, n3, n4]
    with n1+n2+n3+n4 = 14.
    """
    r1, r2, r3, r4, th1, th2, th3, th4, n1, n2, n3, n4 = params
    n1 = int(round(n1)); n2 = int(round(n2)); n3 = int(round(n3)); n4 = int(round(n4))

    # Ensure exactly 14 points
    while n1 + n2 + n3 + n4 > 14:
        if n4 >= 1:
            n4 -= 1
        elif n3 >= 1:
            n3 -= 1
        elif n2 >= 1:
            n2 -= 1
        else:
            n1 -= 1
    while n1 + n2 + n3 + n4 < 14:
        n4 += 1

    pts = []
    if n1 > 0:
        a = th1 + np.linspace(0, 2*np.pi, n1, endpoint=False)
        pts.append(np.column_stack((r1*np.cos(a), r1*np.sin(a))))
    if n2 > 0:
        a = th2 + np.linspace(0, 2*np.pi, n2, endpoint=False)
        pts.append(np.column_stack((r2*np.cos(a), r2*np.sin(a))))
    if n3 > 0:
        a = th3 + np.linspace(0, 2*np.pi, n3, endpoint=False)
        pts.append(np.column_stack((r3*np.cos(a), r3*np.sin(a))))
    if n4 > 0:
        a = th4 + np.linspace(0, 2*np.pi, n4, endpoint=False)
        pts.append(np.column_stack((r4*np.cos(a), r4*np.sin(a))))

    if not pts:
        return np.zeros((N, 2))

    result = np.vstack(pts)
    if len(result) < N:
        extra = np.zeros((N - len(result), 2))
        result = np.vstack([result, extra])
    elif len(result) > N:
        result = result[:N]
    return result

def _neg_ring_score(params):
    try:
        pts = _ring_configuration(params)
        return -compute_score(pts)
    except Exception:
        return 0.0

def _heptagon_plus_seven(params):
    """
    Structural family: 7 points form a regular heptagon (radius R, angle offset phi),
    and 7 additional points are constrained to the heptagon's 7 symmetry axes
    (angles phi + k*pi/7), each with its own radius s_k >= 0.
    params layout: [R, phi, s0, s1, ..., s6]
    """
    R, phi = params[0], params[1]
    s = np.maximum(params[2:], 0.0)
    if len(s) != 7:
        # Fallback: pad/trim
        s = np.resize(s, 7)

    # Heptagon vertices
    a_hep = phi + np.linspace(0, 2*np.pi, 7, endpoint=False)
    p_hep = np.column_stack((R*np.cos(a_hep), R*np.sin(a_hep)))

    # Interior points on symmetry axes
    a_ax = phi + np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/7  # alternate axes
    p_ax = np.column_stack((s*np.cos(a_ax), s*np.sin(a_ax)))

    return np.vstack((p_hep, p_ax))

def _neg_heptagon_score(params):
    try:
        pts = _heptagon_plus_seven(params)
        return -compute_score(pts)
    except Exception:
        return 0.0

def construct():
    start_time = time.time()
    deadline = 142.0  # Safety margin before 150s kill

    best_points = None
    best_score = -float('inf')
    rng = np.random.default_rng(42)

    def remaining():
        return deadline - (time.time() - start_time)

    def try_update(points):
        nonlocal best_points, best_score
        if points is None:
            return 0.0
        try:
            s = compute_score(points)
        except Exception:
            s = 0.0
        if s > best_score:
            best_score = s
            best_points = points.copy()
        return s

    # ---------- Phase 1: Global search in compact parameterizations ----------
    # 1a: 4-ring parameterization via differential evolution
    bounds_ring = [
        (0.3, 1.5),  # r1
        (0.1, 1.0),  # r2
        (0.0, 0.7),  # r3
        (0.0, 0.4),  # r4
        (0, 2*np.pi),  # th1
        (0, 2*np.pi),  # th2
        (0, 2*np.pi),  # th3
        (0, 2*np.pi),  # th4
        (1, 14),  # n1
        (1, 14),  # n2
        (1, 14),  # n3
        (1, 14),  # n4
    ]

    ring_inits = []
    for restart in range(3):
        if remaining() < 45:
            break
        try:
            res = differential_evolution(
                _neg_ring_score,
                bounds_ring,
                seed=int(rng.integers(0, 2**31)),
                maxiter=60,
                tol=1e-6,
                popsize=10,
                mutation=(0.3, 1.2),
                recombination=0.9,
                polish=True,
                init='latinhypercube',
            )
            if res is not None and res.x is not None:
                pts = _ring_configuration(res.x)
                try_update(pts)
                ring_inits.append(pts)
        except Exception:
            pass

    # 1b: Heptagon + 7-axis parameterization via differential evolution
    bounds_hep = [(0.5, 1.5), (0, 2*np.pi)] + [(0.0, 1.0)] * 7
    for restart in range(2):
        if remaining() < 35:
            break
        try:
            res = differential_evolution(
                _neg_heptagon_score,
                bounds_hep,
                seed=int(rng.integers(0, 2**31)),
                maxiter=50,
                tol=1e-6,
                popsize=10,
                mutation=(0.3, 1.2),
                recombination=0.9,
                polish=True,
                init='latinhypercube',
            )
            if res is not None and res.x is not None:
                pts = _heptagon_plus_seven(res.x)
                try_update(pts)
                ring_inits.append(pts)
        except Exception:
            pass

    # ---------- Phase 2: Build diverse full-space starts ----------
    starts = []
    if best_points is not None:
        starts.append(best_points)

    # Add jittered versions of best
    if best_points is not None:
        for _ in range(5):
            if remaining() < 25:
                break
            starts.append(best_points + rng.normal(0, 0.08, (N, 2)))

    # Structured starts
    # Regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    starts.append(np.column_stack((np.cos(angles), np.sin(angles))))

    # Two rings 7+7
    a_out = np.linspace(0, 2*np.pi, 7, endpoint=False)
    a_in = a_out + np.pi/7
    for ratio in [0.45, 0.55, 0.65]:
        starts.append(np.vstack((
            np.column_stack((np.cos(a_out), np.sin(a_out))),
            np.column_stack((ratio*np.cos(a_in), ratio*np.sin(a_in)))
        )))

    # Three rings 5+5+4
    a1 = np.linspace(0, 2*np.pi, 5, endpoint=False)
    a2 = a1 + np.pi/5
    a3 = np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/4
    for r2, r3 in [(0.6, 0.25), (0.55, 0.3), (0.65, 0.2)]:
        starts.append(np.vstack((
            np.column_stack((np.cos(a1), np.sin(a1))),
            np.column_stack((r2*np.cos(a2), r2*np.sin(a2))),
            np.column_stack((r3*np.cos(a3), r3*np.sin(a3)))
        )))

    # Four rings 4+4+3+3
    a1 = np.linspace(0, 2*np.pi, 4, endpoint=False)
    a2 = a1 + np.pi/4
    a3 = np.linspace(0, 2*np.pi, 3, endpoint=False) + np.pi/3
    a4 = a3 + np.pi/3
    for r2, r3, r4 in [(0.65, 0.35, 0.1), (0.6, 0.4, 0.15), (0.7, 0.3, 0.05)]:
        starts.append(np.vstack((
            np.column_stack((np.cos(a1), np.sin(a1))),
            np.column_stack((r2*np.cos(a2), r2*np.sin(a2))),
            np.column_stack((r3*np.cos(a3), r3*np.sin(a3))),
            np.column_stack((r4*np.cos(a4), r4*np.sin(a4)))
        )))

    # Random in disk
    for _ in range(6):
        if remaining() < 20:
            break
        r = np.sqrt(rng.random(N))
        th = rng.random(N) * 2 * np.pi
        starts.append(np.column_stack((r * np.cos(th), r * np.sin(th))))

    # Include any good ring/heptagon configs we found
    for p in ring_inits[:4]:
        starts.append(p)

    # Evaluate and sort starts
    scored = []
    for p in starts:
        try:
            s = compute_score(p)
        except Exception:
            s = 0.0
        scored.append((s, p))
        try_update(p)
    scored.sort(key=lambda x: x[0], reverse=True)

    # ---------- Phase 3: Full 2D refinement from top starts ----------
    # 3a: L-BFGS-B with smoothed objective (tau schedule)
    for init_score, init_points in scored[:8]:
        if remaining() < 20:
            break
        x0 = init_points.flatten()
        # Coarse to fine smoothing
        for tau in [0.05, 0.02, 0.005, 0.001]:
            if remaining() < 10:
                break
            try:
                res = minimize(
                    lambda x: _smooth_min_score(x, tau),
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 40},
                )
                if res.x is not None:
                    pts = res.x.reshape((N, 2))
                    try_update(pts)
                    x0 = res.x
            except Exception:
                pass

    # 3b: Nelder-Mead from top starts (robust local)
    for init_score, init_points in scored[:6]:
        if remaining() < 15:
            break
        try:
            res = minimize(
                neg_score,
                init_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 400, 'xatol': 1e-9, 'ftol': 1e-11},
            )
            if res.x is not None:
                try_update(res.x.reshape((N, 2)))
        except Exception:
            pass

    # ---------- Phase 4: Basin hopping around current best ----------
    if best_points is not None and remaining() > 25:
        for step in [0.35, 0.55, 0.8]:
            if remaining() < 15:
                break
            x_cur = best_points.flatten()
            # Local NM
            try:
                res = minimize(neg_score, x_cur, method='Nelder-Mead',
                               options={'maxiter': 300, 'xatol': 1e-9, 'ftol': 1e-11})
                x_cur = res.x
            except Exception:
                pass
            # Perturb and re-localize
            for _ in range(4):
                if remaining() < 10:
                    break
                pert = rng.normal(0, step, (N, 2)).flatten()
                x_pert = x_cur + pert
                try:
                    res2 = minimize(neg_score, x_pert, method='Nelder-Mead',
                                    options={'maxiter': 150, 'xatol': 1e-9, 'ftol': 1e-11})
                    if res2.fun < -best_score:
                        x_cur = res2.x
                        try_update(x_cur.reshape((N, 2)))
                except Exception:
                    pass
            try_update(x_cur.reshape((N, 2)))

    # ---------- Phase 5: Final polishing ----------
    if best_points is not None and remaining() > 15:
        # Smooth L-BFGS-B with very small tau
        try:
            res = minimize(
                lambda x: _smooth_min_score(x, 1e-4),
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 40},
            )
            if res.x is not None:
                try_update(res.x.reshape((N, 2)))
        except Exception:
            pass

    if best_points is not None and remaining() > 10:
        # Nelder-Mead tight
        try:
            res = minimize(neg_score, best_points.flatten(), method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-11, 'ftol': 1e-13})
            if res.x is not None:
                try_update(res.x.reshape((N, 2)))
        except Exception:
            pass

    if best_points is not None and remaining() > 6:
        # Powell final
        try:
            res = minimize(neg_score, best_points.flatten(), method='Powell',
                           options={'maxiter': 2000, 'xtol': 1e-11, 'ftol': 1e-13})
            if res.x is not None:
                try_update(res.x.reshape((N, 2)))
        except Exception:
            pass

    # Fallback
    if best_points is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        return np.column_stack((np.cos(angles), np.sin(angles)))
    return best_points
