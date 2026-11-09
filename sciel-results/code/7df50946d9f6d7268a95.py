import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def make_grid(step=0.01):
    xs = np.arange(-3, 1.5, step)
    ys = np.arange(-1.5, 1.5, step)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    return X.ravel(), Y.ravel()

def inside_base(u, v):
    return ((u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)) | \
           ((u >= -3) & (u <= 1) & (v >= 0) & (v <= 1))

def densify(path):
    full = np.vstack([[[0.0, 0.0, 0.0]], path, [[-2.0, 0.0, 90.0]]])
    result = [full[0]]
    for i in range(len(full) - 1):
        p1, p2 = full[i], full[i + 1]
        da = abs(p2[2] - p1[2])
        m = max(1, int(np.ceil(da)))
        for k in range(1, m):
            t = k / m
            result.append(p1 * (1 - t) + p2 * t)
        result.append(p2)
    return np.array(result)

def sofa_count(path, X, Y):
    full = densify(path)
    n = X.shape[0]
    mask = np.ones(n, dtype=bool)
    for x, y, a_deg in full:
        a = np.deg2rad(a_deg)
        ca, sa = np.cos(a), np.sin(a)
        dx = X - x
        dy = Y - y
        u = ca * dx + sa * dy
        v = -sa * dx + ca * dy
        mask &= inside_base(u, v)
        if not mask.any():
            return 0
    return int(mask.sum())

def moving_sofa_2d() -> np.ndarray:
    t0 = time.time()
    deadline = 570.0

    def elapsed():
        return time.time() - t0

    # Fallback path (seed)
    seed_path = np.array([
        [-0.1, 0.1, 15.0], [-0.2, 0.2, 20.0], [-0.3, 0.3, 34.0],
        [-0.3, 0.3, 40.0], [-0.4, 0.3, 50.0], [-0.5, 0.2, 55.0],
        [-0.6, 0.2, 75.0], [-0.7, 0.3, 86.0],
    ], dtype=np.float64)

    n = 12
    angles = np.linspace(5.0, 85.0, n)

    def path_from_xy(xy):
        xy = np.asarray(xy, dtype=np.float64).reshape(n, 2)
        return np.column_stack([xy[:, 0], xy[:, 1], angles])

    # Coarse grid
    Xc, Yc = make_grid(0.02)

    def neg_area_coarse(xy):
        return -sofa_count(path_from_xy(xy), Xc, Yc)

    # Phase 1: Bezier (4 params)
    def bezier_path(params):
        c1x, c1y, c2x, c2y = params
        t = np.linspace(0, 1, n)
        P0 = np.array([0.0, 0.0])
        P1 = np.array([c1x, c1y])
        P2 = np.array([c2x, c2y])
        P3 = np.array([-2.0, 0.0])
        pts = np.array([(1 - ti) ** 3 * P0 + 3 * (1 - ti) ** 2 * ti * P1 +
                        3 * (1 - ti) * ti ** 2 * P2 + ti ** 3 * P3 for ti in t])
        return np.column_stack([pts[:, 0], pts[:, 1], angles])

    def neg_area_bezier(params):
        return -sofa_count(bezier_path(params), Xc, Yc)

    best_xy = None
    best_val = np.inf

    # Try a few Bezier starts
    for x0b in [[0.0, 0.3, -1.0, 0.3], [0.5, 0.0, -1.5, 1.0], [-0.5, 1.0, -1.5, 0.0]]:
        if elapsed() > deadline - 200:
            break
        try:
            res = minimize(neg_area_bezier, x0b, method='Nelder-Mead',
                           options={'maxiter': 800, 'xatol': 1e-4, 'fatol': 1e-6})
            if res.fun < best_val:
                best_val = res.fun
                bp = bezier_path(res.x)
                best_xy = bp[:, :2].ravel()
        except Exception:
            pass

    if best_xy is None:
        return seed_path

    # Phase 2: Full param DE
    lb = np.tile([-2.5, -1.0], n)
    ub = np.tile([0.5, 2.0], n)
    bounds = list(zip(lb, ub))

    try:
        if elapsed() < deadline - 150:
            res_de = differential_evolution(neg_area_coarse, bounds,
                                            maxiter=50, tol=1e-4, seed=42,
                                            polish=False, popsize=12,
                                            init='sobol')
            if res_de.fun < best_val:
                best_val = res_de.fun
                best_xy = res_de.x
    except Exception:
        pass

    # Phase 2b: NM polish coarse
    if elapsed() < deadline - 100:
        try:
            res_nm = minimize(neg_area_coarse, best_xy, method='Nelder-Mead',
                              options={'maxiter': 1500, 'xatol': 1e-5, 'fatol': 1e-7})
            if res_nm.fun < best_val:
                best_val = res_nm.fun
                best_xy = res_nm.x
        except Exception:
            pass

    path = path_from_xy(best_xy)

    # Phase 3: Fine grid polish
    if elapsed() < deadline - 30:
        Xf, Yf = make_grid(0.01)

        def neg_area_fine(xy):
            return -sofa_count(path_from_xy(xy), Xf, Yf)

        try:
            res_f = minimize(neg_area_fine, best_xy, method='Nelder-Mead',
                             options={'maxiter': 800, 'xatol': 1e-6, 'fatol': 1e-8})
            if res_f.fun < neg_area_fine(best_xy):
                best_xy = res_f.x
            path = path_from_xy(best_xy)
        except Exception:
            pass

    # Validate
    if not (path.shape == (n, 3) and np.all(np.isfinite(path))):
        return seed_path

    return path
