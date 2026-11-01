import numpy as np
import time
from scipy.optimize import minimize, basinhopping
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

def neg_score(x):
    points = x.reshape((N, 2))
    return -compute_score(points)

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    best_points = None
    best_score = -float('inf')
    rng = np.random.default_rng(12345)

    def try_update(points):
        nonlocal best_points, best_score
        s = compute_score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
        return s

    def remaining():
        return deadline - (time.time() - start_time)

    # --- Generate diverse initial points ---
    inits = []

    # 1. Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    p_reg = np.column_stack((np.cos(angles), np.sin(angles)))
    inits.append(p_reg)

    # 2. Two concentric rings: 7 outer, 7 inner with offset
    a_out = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    a_in = a_out + np.pi / 7
    for ratio in [0.4, 0.5, 0.6, 0.7, 0.8]:
        p_out = np.column_stack((np.cos(a_out), np.sin(a_out)))
        p_in = np.column_stack((ratio * np.cos(a_in), ratio * np.sin(a_in)))
        inits.append(np.vstack((p_out, p_in)))

    # 3. Three rings: 5, 5, 4
    r1, r2, r3 = 1.0, 0.65, 0.35
    a1 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    a2 = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi / 5
    a3 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
    inits.append(np.vstack((
        np.column_stack((r1 * np.cos(a1), r1 * np.sin(a1))),
        np.column_stack((r2 * np.cos(a2), r2 * np.sin(a2))),
        np.column_stack((r3 * np.cos(a3), r3 * np.sin(a3)))
    )))

    # 4. Ellipse-based
    for rx, ry in [(1.2, 0.8), (1.5, 0.65), (1.3, 0.75), (1.4, 0.7)]:
        a = np.linspace(0, 2 * np.pi, N, endpoint=False)
        inits.append(np.column_stack((rx * np.cos(a), ry * np.sin(a))))

    # 5. Random in disk
    for _ in range(10):
        r = np.sqrt(rng.random(N))
        th = rng.random(N) * 2 * np.pi
        inits.append(np.column_stack((r * np.cos(th), r * np.sin(th))))

    # 6. Jittered regular polygon
    for _ in range(8):
        inits.append(p_reg + rng.normal(0, 0.15, (N, 2)))

    # 7. Random in square (different hull shape)
    for _ in range(5):
        inits.append(rng.uniform(-1, 1, (N, 2)))

    # 8. Two rings with different counts: 6+8, 8+6
    for n_out, n_in in [(6, 8), (8, 6)]:
        a_o = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
        a_i = np.linspace(0, 2 * np.pi, n_in, endpoint=False) + np.pi / n_in
        for ratio in [0.5, 0.7]:
            p_o = np.column_stack((np.cos(a_o), np.sin(a_o)))
            p_i = np.column_stack((ratio * np.cos(a_i), ratio * np.sin(a_i)))
            inits.append(np.vstack((p_o, p_i)))

    # Evaluate all inits
    for p in inits:
        try_update(p)

    # Sort by score, keep top for refinement
    scored = [(compute_score(p), p) for p in inits]
    scored.sort(key=lambda x: x[0], reverse=True)

    # Phase 1: BasinHopping from top initial points
    for init_score, init_points in scored[:6]:
        if remaining() < 15:
            break
        x0 = init_points.flatten()
        try:
            res = basinhopping(
                neg_score, x0,
                method='Nelder-Mead',
                options={'maxiter': 400, 'xatol': 1e-7, 'fatol': 1e-9},
                minsteps=4, T=1.0, stepsize=0.5,
                interval=6, disp=False
            )
            new_p = res.x.reshape((N, 2))
            try_update(new_p)
        except Exception:
            pass

    # Phase 2: Powell refinement from best
    if best_points is not None and remaining() > 20:
        x0 = best_points.flatten()
        try:
            res = minimize(neg_score, x0, method='Powell',
                           options={'maxiter': 600, 'xtol': 1e-8, 'ftol': 1e-10})
            try_update(res.x.reshape((N, 2)))
        except Exception:
            pass

    # Phase 3: Basin hopping with varied params from best
    if best_points is not None and remaining() > 25:
        for step, T in [(0.3, 0.8), (0.5, 1.2), (0.7, 1.5), (0.4, 0.6)]:
            if remaining() < 12:
                break
            x0 = best_points.flatten()
            try:
                res = basinhopping(
                    neg_score, x0,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9},
                    minsteps=4, T=T, stepsize=step,
                    interval=8, disp=False
                )
                try_update(res.x.reshape((N, 2)))
            except Exception:
                pass

    # Phase 4: Final tight Nelder-Mead polish
    if best_points is not None and remaining() > 10:
        x0 = best_points.flatten()
        try:
            res = minimize(neg_score, x0, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-11})
            try_update(res.x.reshape((N, 2)))
        except Exception:
            pass

    # Phase 5: Another basin hopping + Powell if time permits
    if best_points is not None and remaining() > 15:
        x0 = best_points.flatten()
        try:
            res = basinhopping(
                neg_score, x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10},
                minsteps=5, T=0.5, stepsize=0.25,
                interval=6, disp=False
            )
            try_update(res.x.reshape((N, 2)))
        except Exception:
            pass

    if best_points is not None and remaining() > 8:
        x0 = best_points.flatten()
        try:
            res = minimize(neg_score, x0, method='Powell',
                           options={'maxiter': 1000, 'xtol': 1e-9, 'ftol': 1e-11})
            try_update(res.x.reshape((N, 2)))
        except Exception:
            pass

    # Phase 6: Final ultra-tight polish
    if best_points is not None and remaining() > 5:
        x0 = best_points.flatten()
        try:
            res = minimize(neg_score, x0, method='Nelder-Mead',
                           options={'maxiter': 8000, 'xatol': 1e-10, 'fatol': 1e-12})
            try_update(res.x.reshape((N, 2)))
        except Exception:
            pass

    if best_points is None:
        return p_reg
    return best_points
