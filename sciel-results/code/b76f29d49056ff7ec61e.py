import math
import numpy as np
import time
from scipy.optimize import linprog, minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 6.0
EPS = 1e-9
PI = math.pi


def _normals(beta):
    # 6 outward normals of the outer hexagon for rotation beta
    return np.array([
        [math.cos(beta + PI / 6.0 + k * PI / 3.0),
         math.sin(beta + PI / 6.0 + k * PI / 3.0)]
        for k in range(6)
    ])


def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    candidates = []

    # ---- Hexagonal clusters (center + neighbors) ----
    candidates.append([
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)])
    candidates.append([
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)])
    candidates.append([
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (-1, 3), (1, -2)])
    candidates.append([
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, 2)])
    candidates.append([
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, -1)])
    candidates.append([
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (1, 2), (-1, 2), (0, 3)])
    candidates.append([
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, 1)])
    candidates.append([
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)])
    candidates.append([
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, -2)])
    candidates.append([
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)])

    # ---- Rectangular / block-ish packs ----
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)])
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)])
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)])
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)])
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)])
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3)])
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4)])
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1), (0, -1)])
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 2)])
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2), (3, 2)])
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 3)])
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2)])
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3)])
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (0, 4)])
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)])

    # ---- Compact / asymmetric packs (diversity) ----
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (1, 4)])
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)])
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4)])
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)])
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2),
        (0, 3), (3, 3)])
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (0, 3), (1, 3),
        (2, 3)])
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3),
        (2, 3)])
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (3, 3)])
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (0, 4)])
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3), (2, 3)])
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3), (2, 3),
        (0, 4), (1, 4)])
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3),
        (1, 4)])
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4), (1, 4)])
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3),
        (0, 4), (1, 4)])
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3), (2, 3),
        (1, 4), (2, 4)])
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4), (1, 4)])
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (1, 4), (2, 4)])
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3), (3, 3)])
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (1, 4), (2, 4)])
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)])
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)])

    # De-duplicate candidates (as sets) to avoid redundant optimization
    seen = set()
    uniq_candidates = []
    for c in candidates:
        key = frozenset(c)
        if key not in seen:
            seen.add(key)
            uniq_candidates.append(c)
    candidates = uniq_candidates

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def calc_S_for_beta(pts, beta):
        """Exact LP: minimize D s.t. outer hexagon of rotation beta encloses pts."""
        normals = _normals(beta)
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        for nx, ny in normals:
            for px, py in pts:
                proj = px * nx + py * ny
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-proj)
                A_ub.append([nx, ny, -1.0])
                b_ub.append(proj)
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(None, None), (None, None), (0, None)]
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = res.x[2]
                cx, cy = res.x[0], res.x[1]
                S = (D + APOTHEM) / APOTHEM
                return S, (cx, cy), D
            return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    def smooth_calc_S(pts, beta, cx, cy, temperature=0.1):
        normals = _normals(beta)
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])
        max_val = np.max(all_projs)
        if max_val > 100:
            return float('inf'), (cx, cy), float('inf')
        shifted = (all_projs - max_val) / temperature
        shifted = np.clip(shifted, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
        D = temperature * log_sum
        S = (D + APOTHEM) / APOTHEM
        return S, (cx, cy), D

    def exact_max_proj(pts, beta, cx, cy):
        normals = _normals(beta)
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        return np.max(np.abs(projs))

    def newton_polish_center(pts, beta, cx, cy, max_iter=200, tol=1e-13):
        """2D Newton / gradient-descent polish of the center at fixed beta.

        f(c) = max_k max_i (n_k . (p_i - c))  (a convex piecewise-linear function).
        Subgradient = -sum over active (k,i) of n_k.  We do projected subgradient
        with backtracking, which converges to the minimizer of this convex function.
        """
        normals = _normals(beta)  # (6,2)
        cx, cy = float(cx), float(cy)
        for it in range(max_iter):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            # active constraints
            projs = (pts - np.array([cx, cy])) @ normals.T  # (n,6)
            neg_projs = -projs
            allp = np.concatenate([projs.ravel(), neg_projs.ravel()])
            m = np.max(allp)
            active = allp >= m - 1e-12
            # build subgradient: for each active (k,i) contribution of -n_k (since
            # d/dc of n.(p-c) = -n)
            # projs.ravel() order: for each point i, k=0..5 ; neg similar
            grad = np.zeros(2)
            # projs part: index (i*6 + k)
            idx = np.where(active)[0]
            for a in idx:
                if a < 6 * len(pts):
                    i = a // 6
                    k = a % 6
                    grad -= normals[k]
                else:
                    a2 = a - 6 * len(pts)
                    i = a2 // 6
                    k = a2 % 6
                    # neg_proj = -n.(p-c); d/dc = +n
                    grad += normals[k]
            if np.linalg.norm(grad) < 1e-15:
                break
            # backtracking line search on f
            f0 = m
            step = 1.0
            improved = False
            for _ in range(40):
                nx_ = cx - step * grad[0]
                ny_ = cy - step * grad[1]
                f1 = exact_max_proj(pts, beta, nx_, ny_)
                if f1 < f0 - 1e-15:
                    cx, cy = nx_, ny_
                    improved = True
                    break
                step *= 0.5
            if not improved:
                break
            if f0 - exact_max_proj(pts, beta, cx, cy) < tol:
                pass
        return cx, cy

    def optimize_shape(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None

        cx0 = float(np.mean(pts[:, 0]))
        cy0 = float(np.mean(pts[:, 1]))

        # coarse grid over beta with smooth surrogate at centroid
        best_smooth_S = float('inf')
        best_smooth_beta = 0.0
        for i in range(97):
            beta = i * (PI / 6) / 96
            S_s, _, _ = smooth_calc_S(pts, beta, cx0, cy0, temperature=0.1)
            if S_s < best_smooth_S:
                best_smooth_S = S_s
                best_smooth_beta = beta

        # smooth-surrogate Nelder-Mead from multiple starts
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                temperatures = [0.2, 0.08, 0.03, 0.01]
                start_betas = [best_smooth_beta, 0.0, PI / 12, PI / 6,
                               best_smooth_beta + 0.03, best_smooth_beta - 0.03,
                               best_smooth_beta + 0.06, best_smooth_beta - 0.06]
                for temp in temperatures:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    for sb in start_betas:
                        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                            break

                        def smooth_obj(params, t=temp):
                            cx, cy, beta = params
                            beta = ((beta % (PI / 6)) + PI / 6) % (PI / 6)
                            S, _, _ = smooth_calc_S(pts, beta, cx, cy, temperature=t)
                            return S

                        init = [cx0, cy0, sb]
                        res = minimize(smooth_obj, init, method='Nelder-Mead',
                                       options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12})
                        if res.fun < 1e10:
                            cx_r, cy_r, beta_r = res.x
                            beta_r = ((beta_r % (PI / 6)) + PI / 6) % (PI / 6)
                            S_exact, c_exact, _ = calc_S_for_beta(pts, beta_r)
                            if S_exact < best_S:
                                best_S = S_exact
                                best_beta = beta_r
                                best_center = c_exact
        except Exception:
            pass

        if best_center is None:
            S_exact, c_exact, _ = calc_S_for_beta(pts, best_smooth_beta)
            if S_exact < best_S:
                best_S = S_exact
                best_beta = best_smooth_beta
                best_center = c_exact

        # Exact objective polish (Nelder-Mead + SLSQP) with diverse starts
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN and best_center is not None:
                def objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (PI / 6)) + PI / 6) % (PI / 6)
                    max_proj = exact_max_proj(pts, beta, cx, cy)
                    return (max_proj + APOTHEM) / APOTHEM

                d = 0.01
                polish_starts = [
                    [best_center[0], best_center[1], best_beta],
                    [best_center[0], best_center[1], best_beta + d],
                    [best_center[0], best_center[1], best_beta - d],
                    [best_center[0], best_center[1], best_beta + 2 * d],
                    [best_center[0], best_center[1], best_beta - 2 * d],
                    [best_center[0] + d, best_center[1], best_beta],
                    [best_center[0] - d, best_center[1], best_beta],
                    [best_center[0], best_center[1] + d, best_beta],
                    [best_center[0], best_center[1] - d, best_beta],
                    [best_center[0] + d, best_center[1] + d, best_beta],
                    [best_center[0] - d, best_center[1] - d, best_beta],
                ]
                for init in polish_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    res = minimize(objective, init, method='Nelder-Mead',
                                   options={'maxiter': 20000, 'xatol': 1e-14, 'fatol': 1e-14})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_center = (res.x[0], res.x[1])
                        best_beta = ((res.x[2] % (PI / 6)) + PI / 6) % (PI / 6)

                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    try:
                        res2 = minimize(objective, [best_center[0], best_center[1], best_beta],
                                        method='SLSQP', options={'maxiter': 4000, 'ftol': 1e-15})
                        if res2.fun < best_S:
                            best_S = res2.fun
                            best_center = (res2.x[0], res2.x[1])
                            best_beta = ((res2.x[2] % (PI / 6)) + PI / 6) % (PI / 6)
                    except Exception:
                        pass
        except Exception:
            pass

        # High-precision Newton center polish at the optimal beta (inspiration idea)
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN and best_center is not None:
                cx, cy = newton_polish_center(pts, best_beta, best_center[0], best_center[1])
                max_proj = exact_max_proj(pts, best_beta, cx, cy)
                S_new = (max_proj + APOTHEM) / APOTHEM
                if S_new < best_S:
                    best_S = S_new
                    best_center = (cx, cy)
                # also try a couple of nearby betas with Newton polish
                for db in (0.005, -0.005, 0.01, -0.01):
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    beta2 = ((best_beta + db) % (PI / 6) + PI / 6) % (PI / 6)
                    cx2, cy2 = newton_polish_center(pts, beta2, cx, cy)
                    mp2 = exact_max_proj(pts, beta2, cx2, cy2)
                    S2 = (mp2 + APOTHEM) / APOTHEM
                    if S2 < best_S:
                        best_S = S2
                        best_center = (cx2, cy2)
                        best_beta = beta2
        except Exception:
            pass

        return best_S, best_beta, best_center

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    def quick_estimate(axial_list):
        pts = axial_to_cartesian(axial_list)
        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])
        max_proj = 0
        for k in range(6):
            ang = PI / 6.0 + k * PI / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
            max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))
        return (max_proj + APOTHEM) / APOTHEM

    estimated_scores = [(quick_estimate(ax), ax) for ax in candidates]
    estimated_scores.sort(key=lambda x: x[0])

    for rank, (est, axial) in enumerate(estimated_scores):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        if rank > 8 and est > best_global_S * 1.05:
            continue
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = axial_to_cartesian(candidates[0])
        best_global_center = (float(np.mean(best_global_pts[:, 0])),
                              float(np.mean(best_global_pts[:, 1])))
        best_global_beta = 0.0
        max_proj = 0
        for k in range(6):
            ang = PI / 6.0 + k * PI / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
            max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))
        best_global_S = (max_proj + APOTHEM) / APOTHEM

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)

    # tiny safety scale so inner hexagons stay strictly inside
    SCALE = 1.0 + 1e-10
    pts_scaled = np.array([
        (cx + (x - cx) * SCALE, cy + (y - cy) * SCALE) for x, y in best_global_pts
    ])

    max_proj = 0
    for k in range(6):
        ang = best_global_beta + PI / 6.0 + k * PI / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (pts_scaled[:, 0] - cx) * nx + (pts_scaled[:, 1] - cy) * ny
        max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))
    D_scaled = max_proj
    S_scaled = (D_scaled + APOTHEM) / APOTHEM

    inner = [[float(x), float(y), float(angle_deg)] for x, y in pts_scaled]
    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)
    return inner, center, side, rotation
