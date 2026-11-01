import numpy as np
import time
from scipy.optimize import minimize, basinhopping, differential_evolution
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
    for ratio in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        p_out = np.column_stack((np.cos(a_out), np.sin(a_out)))
        p_in = np.column_stack((ratio * np.cos(a_in), ratio * np.sin(a_in)))
        inits.append(np.vstack((p_out, p_in)))

    # 3. Three rings: 5, 5, 4
    for r1, r2, r3 in [(1.0, 0.6, 0.3), (1.0, 0.7, 0.4), (1.0, 0.5, 0.25)]:
        a1 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        a2 = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi / 5
        a3 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
        inits.append(np.vstack((
            np.column_stack((r1 * np.cos(a1), r1 * np.sin(a1))),
            np.column_stack((r2 * np.cos(a2), r2 * np.sin(a2))),
            np.column_stack((r3 * np.cos(a3), r3 * np.sin(a3)))
        )))

    # 4. Ellipse-based
    for rx, ry in [(1.2, 0.8), (1.5, 0.65), (1.3, 0.75), (1.4, 0.7), (1.6, 0.6)]:
        a = np.linspace(0, 2 * np.pi, N, endpoint=False)
        inits.append(np.column_stack((rx * np.cos(a), ry * np.sin(a))))

    # 5. Random in disk
    for _ in range(15):
        r = np.sqrt(rng.random(N))
        th = rng.random(N) * 2 * np.pi
        inits.append(np.column_stack((r * np.cos(th), r * np.sin(th))))

    # 6. Jittered regular polygon
    for _ in range(10):
        inits.append(p_reg + rng.normal(0, 0.2, (N, 2)))

    # 7. Random in square (different hull shape)
    for _ in range(8):
        inits.append(rng.uniform(-1, 1, (N, 2)))

    # 8. Two rings with different counts: 6+8, 8+6
    for n_out, n_in in [(6, 8), (8, 6), (5, 9), (9, 5)]:
        a_o = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
        a_i = np.linspace(0, 2 * np.pi, n_in, endpoint=False) + np.pi / n_in
        for ratio in [0.4, 0.5, 0.6, 0.7, 0.8]:
            p_o = np.column_stack((np.cos(a_o), np.sin(a_o)))
            p_i = np.column_stack((ratio * np.cos(a_i), ratio * np.sin(a_i)))
            inits.append(np.vstack((p_o, p_i)))

    # 9. Square-based configurations
    for scale in [0.8, 1.0, 1.2]:
        # Points on square boundary
        sq = np.array([
            [scale, scale], [scale, -scale], [-scale, -scale], [-scale, scale]
        ])
        # Add points along edges
        extra = []
        for i in range(10):
            t = (i + 1) / 11.0
            edge = i % 4
            if edge == 0:
                extra.append([scale * (1 - 2*t), scale])
            elif edge == 1:
                extra.append([scale, scale * (1 - 2*t)])
            elif edge == 2:
                extra.append([scale * (1 - 2*t), -scale])
            else:
                extra.append([-scale, scale * (1 - 2*t)])
        inits.append(np.array(extra))

    # 10. Hexagon-based
    for scale in [0.9, 1.0, 1.1]:
        a_hex = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        p_hex = scale * np.column_stack((np.cos(a_hex), np.sin(a_hex)))
        # Add interior points
        a_int = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi/8
        p_int = 0.5 * scale * np.column_stack((np.cos(a_int), np.sin(a_int)))
        inits.append(np.vstack((p_hex, p_int)))

    # Evaluate all inits
    for p in inits:
        try_update(p)

    # Sort by score, keep top for refinement
    scored = [(compute_score(p), p) for p in inits]
    scored.sort(key=lambda x: x[0], reverse=True)

    # Phase 1: Differential Evolution for global search
    if remaining() > 40:
        try:
            bounds = [(-2, 2)] * (N * 2)
            de_res = differential_evolution(
                neg_score, bounds,
                maxiter=50, popsize=20,
                tol=1e-6, seed=12345,
                polish=True
            )
            new_p = de_res.x.reshape((N, 2))
            try_update(new_p)
        except Exception:
            pass

    # Phase 2: BasinHopping from top initial points
    for init_score, init_points in scored[:8]:
        if remaining() < 20:
            break
        x0 = init_points.flatten()
        try:
            res = basinhopping(
                neg_score, x0,
                method='Nelder-Mead',
                options={'maxiter': 400, 'xatol': 1e-7, 'fatol': 1e-9},
                minsteps=5, T=1.0, stepsize=0.5,
                interval=5, disp=False
            )
            new_p = res.x.reshape((N, 2))
            try_update(new_p)
        except Exception:
            pass

    # Phase 3: Powell refinement from best
    if best_points is not None and remaining() > 25:
        x0 = best_points.flatten()
        try:
            res = minimize(neg_score, x0, method='Powell',
                           options={'maxiter': 800, 'xtol': 1e-8, 'ftol': 1e-10})
            try_update(res.x.reshape((N, 2)))
        except Exception:
            pass

    # Phase 4: Basin hopping with varied params from best
    if best_points is not None and remaining() > 30:
        for step, T in [(0.3, 0.8), (0.5, 1.2), (0.7, 1.5), (0.4, 0.6), (0.6, 1.0)]:
            if remaining() < 15:
                break
            x0 = best_points.flatten()
            try:
                res = basinhopping(
                    neg_score, x0,
                    method='Nelder-Mead',
                    options={'maxiter': 400, 'xatol': 1e-7, 'fatol': 1e-9},
                    minsteps=5, T=T, stepsize=step,
                    interval=6, disp=False
                )
                try_update(res.x.reshape((N, 2)))
            except Exception:
                pass

    # Phase 5: Final tight Nelder-Mead polish
    if best_points is not None and remaining() > 12:
        x0 = best_points.flatten()
        try:
            res = minimize(neg_score, x0, method='Nelder-Mead',
                           options={'maxiter': 6000, 'xatol': 1e-9, 'fatol': 1e-11})
            try_update(res.x.reshape((N, 2)))
        except Exception:
            pass

    # Phase 6: Another basin hopping + Powell if time permits
    if best_points is not None and remaining() > 18:
        x0 = best_points.flatten()
        try:
            res = basinhopping(
                neg_score, x0,
                method='Nelder-Mead',
                options={'maxiter': 600, 'xatol': 1e-8, 'fatol': 1e-10},
                minsteps=6, T=0.5, stepsize=0.25,
                interval=5, disp=False
            )
            try_update(res.x.reshape((N, 2)))
        except Exception:
            pass

    if best_points is not None and remaining() > 10:
        x0 = best_points.flatten()
        try:
            res = minimize(neg_score, x0, method='Powell',
                           options={'maxiter': 1500, 'xtol': 1e-9, 'ftol': 1e-11})
            try_update(res.x.reshape((N, 2)))
        except Exception:
            pass

    # Phase 7: Final ultra-tight polish
    if best_points is not None and remaining() > 6:
        x0 = best_points.flatten()
        try:
            res = minimize(neg_score, x0, method='Nelder-Mead',
                           options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12})
            try_update(res.x.reshape((N, 2)))
        except Exception:
            pass

    if best_points is None:
        return p_reg
    return best_points
