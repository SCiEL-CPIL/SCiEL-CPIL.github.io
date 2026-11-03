import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 12.0
EPS = 1e-9


def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    # --- Candidate point sets (axial coordinates) ---
    candidates = []
    candidates.append([(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (1, 1), (-1, 2), (-2, 1), (2, -1)])
    candidates.append([(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (2, 0), (-2, 0), (0, 2), (0, -2)])
    candidates.append([(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2), (-1, 1), (3, 1)])
    candidates.append([(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2)])
    candidates.append([(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2), (0, 3), (1, 3), (2, 3)])
    candidates.append([(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2), (0, 3), (1, 3)])
    candidates.append([(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (2, 0), (-2, 0), (0, 2), (0, -2)])
    candidates.append([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)])
    candidates.append([(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (2, 1), (-2, 2), (-1, 3), (1, -2)])
    candidates.append([(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2), (1, 3), (2, 3)])
    candidates.append([(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (3, 1), (0, 2), (1, 2), (2, 2), (3, 2), (1, 3)])
    candidates.append([(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (0, 2), (1, 2), (0, 3), (1, 3), (0, 4)])
    candidates.append([(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (2, 0), (-2, 0), (0, 2), (1, 2)])
    candidates.append([(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (2, 0), (-2, 0), (1, 1), (-1, -1)])
    candidates.append([(0, 0), (1, 0), (2, 0), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2), (-1, 1), (3, 1), (0, -1)])
    candidates.append([(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (1, 2), (2, 2), (0, 3), (1, 3), (2, 3), (3, 2)])
    candidates.append([(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (2, 1), (1, 2), (-1, 2), (0, 3)])
    candidates.append([(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (1, 2), (2, 2), (3, 2)])
    candidates.append([(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (1, 2), (2, 2), (0, 3), (1, 3), (2, 3), (3, 3)])
    candidates.append([(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (2, 0), (-2, 0), (1, 1), (-1, 1)])
    candidates.append([(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1), (0, 2), (1, 2), (2, 2)])
    candidates.append([(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (3, 1), (0, 2), (1, 2), (2, 2), (1, 3)])
    candidates.append([(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (2, 0), (-2, 0), (0, 2), (0, -2)])
    candidates.append([(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2), (0, 3), (1, 3), (0, 4)])
    candidates.append([(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2), (0, 3), (1, 3), (2, 3)])
    candidates.append([(0, 0), (1, 0), (2, 0), (3, 0), (1, 1), (2, 1), (3, 1), (0, 2), (1, 2), (2, 2), (3, 2), (1, 3), (2, 3)])
    candidates.append([(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (2, 0), (-2, 0), (0, 2), (1, -2)])
    candidates.append([(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3), (1, 4), (2, 4)])
    candidates.append([(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (3, 1), (0, 2), (1, 2), (2, 2), (1, 3), (2, 3)])
    candidates.append([(0, 0), (1, 0), (2, 0), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2), (1, 3), (2, 3), (0, -1), (3, 1)])
    candidates.append([(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1), (0, 2), (1, 2), (2, 2)])
    candidates.append([(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (2, 0), (-2, 0), (1, 1), (-1, -1)])
    candidates.append([(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (1, 2), (2, 2), (0, 3), (1, 3), (2, 3), (1, 4), (2, 4)])
    candidates.append([(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2), (0, 3), (1, 3)])
    candidates.append([(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (1, 2), (2, 2), (0, 3), (1, 3)])
    # Corner-filling templates
    candidates.append([(2, 0), (-2, 0), (0, 2), (0, -2), (2, -2), (-2, 2), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1)])
    candidates.append([(2, 0), (-2, 0), (0, 2), (0, -2), (2, -2), (-2, 2), (1, 0), (-1, 0), (1, 1), (-1, 1), (0, 1)])
    candidates.append([(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (2, 0), (-2, 0), (0, 2), (0, -2)])
    candidates.append([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (-1, 1), (0, 1), (1, 1), (2, 1), (3, 1), (4, 1)])
    candidates.append([(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (0, 3), (1, 3)])
    candidates.append([(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1), (2, 0), (-2, 0), (0, 2), (0, -2), (2, -2)])
    candidates.append([(0, 0), (2, 0), (-2, 0), (0, 2), (0, -2), (2, -2), (-2, 2), (1, 1), (-1, 1), (1, -1), (-1, -1)])
    candidates.append([(2, 0), (-2, 0), (0, 2), (0, -2), (2, -2), (-2, 2), (1, 0), (-1, 0), (0, 1), (1, -1), (-1, 1)])
    candidates.append([(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1), (2, 0), (-2, 0), (0, 2), (0, -2), (1, 1)])
    candidates.append([(2, 0), (-2, 0), (0, 2), (0, -2), (2, -2), (-2, 2), (1, 0), (0, 1), (1, 1), (0, -1), (-1, 0)])

    def calc_S_for_beta(pts, beta):
        """Vectorized LP for minimum enclosing hexagon with given rotation."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None

        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])

        n_pts = len(pts)
        n_constraints = 12 * n_pts
        A_ub = np.zeros((n_constraints, 3))
        b_ub = np.zeros(n_constraints)

        for k in range(6):
            nx, ny = normals[k]
            projs = pts[:, 0] * nx + pts[:, 1] * ny
            start = 2 * k * n_pts
            A_ub[start:start + n_pts, 0] = -nx
            A_ub[start:start + n_pts, 1] = -ny
            A_ub[start:start + n_pts, 2] = -1.0
            b_ub[start:start + n_pts] = -projs
            A_ub[start + n_pts:start + 2 * n_pts, 0] = nx
            A_ub[start + n_pts:start + 2 * n_pts, 1] = ny
            A_ub[start + n_pts:start + 2 * n_pts, 2] = -1.0
            b_ub[start + n_pts:start + 2 * n_pts] = projs

        c_obj = np.array([0.0, 0.0, 1.0])
        bounds = [(None, None), (None, None), (0, None)]

        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = res.x[2]
                cx = res.x[0]
                cy = res.x[1]
                S = (D + APOTHEM) / APOTHEM
                return S, (cx, cy), D
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    def exact_S(pts, cx, cy, beta):
        """Exact computation of S for given center and rotation."""
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        D = np.max(np.abs(projs))
        return (D + APOTHEM) / APOTHEM

    def smooth_calc_S(pts, cx, cy, beta, temperature=0.05):
        """Smooth surrogate using log-sum-exp."""
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])
        max_val = np.max(all_projs)
        if max_val > 200:
            return float('inf')
        shifted = (all_projs - max_val) / temperature
        shifted = np.clip(shifted, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
        D = temperature * log_sum
        return (D + APOTHEM) / APOTHEM

    def newton_polish_center(pts, beta, cx, cy, max_iter=2000):
        """
        Newton-like polish of the center for fixed beta using the exact subgradient.
        D(c) = max_{i,k} |(p_i - c).n_k| is convex piecewise-linear in c.
        Use subgradient descent with line search; near optimum, multiple active
        constraints give a unique minimizer (the LP solution).
        """
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])

        best_cx, best_cy = cx, cy
        best_D = float('inf')

        step = 0.5
        for it in range(max_iter):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            pts_centered = pts - np.array([cx, cy])
            projs = pts_centered @ normals.T  # (N, 6)
            D = np.max(np.abs(projs))
            if D < best_D:
                best_D = D
                best_cx, best_cy = cx, cy

            # Subgradient: average of active normals (with sign)
            active_pos = np.abs(projs) > D - 1e-9
            active_neg = np.abs(projs) > D - 1e-9
            # For each (i,k): if projs[i,k] > D - eps, grad contribution is +n_k
            # if -projs[i,k] > D - eps (i.e. projs[i,k] < -D + eps), grad is -n_k
            # Sum over active, normalize
            grad = np.zeros(2)
            count = 0
            for k in range(6):
                pos_mask = projs[:, k] > D - 1e-9
                neg_mask = projs[:, k] < -D + 1e-9
                n_pos = np.sum(pos_mask)
                n_neg = np.sum(neg_mask)
                count += n_pos + n_neg
                grad[0] += n_pos * normals[k, 0] - n_neg * normals[k, 0]
                grad[1] += n_pos * normals[k, 1] - n_neg * normals[k, 1]

            if count == 0:
                break

            grad = grad / count
            gnorm = np.linalg.norm(grad)
            if gnorm < 1e-10:
                break

            # Line search
            for _ in range(30):
                new_cx = cx - step * grad[0]
                new_cy = cy - step * grad[1]
                D_new = np.max(np.abs((pts - np.array([new_cx, new_cy])) @ normals.T))
                if D_new < D - 1e-12:
                    cx, cy = new_cx, new_cy
                    break
                step *= 0.5
            else:
                break

            # If we're at the LP optimum (multiple active constraints spanning), step can be tiny
            if step < 1e-12:
                break

        # Final exact
        D_final = np.max(np.abs((pts - np.array([best_cx, best_cy])) @ normals.T))
        S_final = (D_final + APOTHEM) / APOTHEM
        return best_cx, best_cy, S_final

    def optimize_shape(pts):
        """Multi-stage optimization for a fixed point configuration."""
        best_S = float('inf')
        best_beta = 0.0
        best_center = None

        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None

        # Coarse grid search over beta
        for i in range(37):
            beta = i * (math.pi / 6) / 36
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center

        # Refine beta with bounded scalar optimization
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0],
                                     bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 300, 'xatol': 1e-12})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass

        # Smooth surrogate optimization with adaptive temperature
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                temperatures = [0.3, 0.1, 0.05, 0.02, 0.01]
                start_betas = [best_beta, best_beta + 0.05, best_beta - 0.05, 0.0, math.pi / 12, math.pi / 6]

                for temp in temperatures:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    for sb in start_betas:
                        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                            break

                        def smooth_obj(params, t=temp):
                            cx, cy, beta = params
                            beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                            return smooth_calc_S(pts, beta, cx, cy, t)

                        init = [best_center[0], best_center[1], sb]
                        res = minimize(smooth_obj, init, method='Nelder-Mead',
                                      options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})

                        if res.fun < 1e10:
                            cx_r, cy_r, beta_r = res.x
                            beta_r = ((beta_r % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                            S_exact, c_exact, _ = calc_S_for_beta(pts, beta_r)
                            if S_exact < best_S:
                                best_S = S_exact
                                best_beta = beta_r
                                best_center = c_exact
        except Exception:
            pass

        # Final high-precision polish using exact objective
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    return exact_S(pts, cx, cy, beta)

                polish_starts = [
                    [best_center[0], best_center[1], best_beta],
                    [best_center[0], best_center[1], best_beta + 0.02],
                    [best_center[0], best_center[1], best_beta - 0.02],
                    [best_center[0] + 0.02, best_center[1], best_beta],
                    [best_center[0], best_center[1] + 0.02, best_beta],
                    [best_center[0] - 0.02, best_center[1], best_beta],
                    [best_center[0], best_center[1] - 0.02, best_beta],
                ]

                for init in polish_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    res = minimize(objective, init, method='Nelder-Mead',
                                  options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-13})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_center = (res.x[0], res.x[1])
                        best_beta = ((res.x[2] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
        except Exception:
            pass

        # Newton center polish (exact subgradient)
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                cx_p, cy_p, S_p = newton_polish_center(pts, best_beta, best_center[0], best_center[1])
                if S_p < best_S:
                    best_S = S_p
                    best_center = (cx_p, cy_p)
        except Exception:
            pass

        return best_S, best_beta, best_center

    def continuous_optimization(initial_pts):
        """
        Optimize the relative positions of the hexagons and the outer rotation.
        """
        N_pts = len(initial_pts)
        x0 = np.concatenate([initial_pts.flatten(), [0.0]])

        def objective(params):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                return float('inf')

            pos = params[:2 * N_pts].reshape(N_pts, 2)
            beta = params[2 * N_pts]

            center = np.mean(pos, axis=0)
            S = exact_S(pos, center[0], center[1], beta)

            penalty = 0.0
            for i in range(N_pts):
                for j in range(i + 1, N_pts):
                    d = np.linalg.norm(pos[i] - pos[j])
                    if d < 2.0:
                        penalty += (2.0 - d) ** 2

            return S + 100.0 * penalty

        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})

            if res.fun < float('inf'):
                pos_opt = res.x[:2 * N_pts].reshape(N_pts, 2)
                beta_opt = res.x[2 * N_pts]

                valid = True
                for i in range(N_pts):
                    for j in range(i + 1, N_pts):
                        if np.linalg.norm(pos_opt[i] - pos_opt[j]) < 2.0 - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break

                if valid:
                    S, center, D = calc_S_for_beta(pos_opt, beta_opt)
                    return S, beta_opt, center, pos_opt
                else:
                    return float('inf'), 0.0, None, None
        except Exception:
            return float('inf'), 0.0, None, None

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    # First pass: Discrete candidates
    for axial in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

    # Second pass: Continuous optimization starting from best discrete and a few others
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        S_c, beta_c, center_c, pts_c = continuous_optimization(best_global_pts)
        if S_c < best_global_S:
            best_global_S = S_c
            best_global_beta = beta_c
            best_global_center = center_c
            best_global_pts = pts_c

        for axial in [candidates[0], candidates[1], candidates[2], candidates[4], candidates[10]]:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            pts = axial_to_cartesian(axial)
            S_disc, _, _ = optimize_shape(pts)
            if S_disc < best_global_S + 0.5:
                S_c, beta_c, center_c, pts_c = continuous_optimization(pts)
                if S_c < best_global_S:
                    best_global_S = S_c
                    best_global_beta = beta_c
                    best_global_center = center_c
                    best_global_pts = pts_c

    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = axial_to_cartesian(candidates[0])
        best_global_center = (float(np.mean(best_global_pts[:, 0])), float(np.mean(best_global_pts[:, 1])))
        best_global_beta = 0.0
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        best_global_S = (max_proj + APOTHEM) / APOTHEM

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)

    # Check validity
    valid_final = True
    for i in range(len(best_global_pts)):
        for j in range(i + 1, len(best_global_pts)):
            if np.linalg.norm(best_global_pts[i] - best_global_pts[j]) < 2.0 - 1e-9:
                valid_final = False
                break
        if not valid_final:
            break

    if not valid_final:
        best_disc_S = float('inf')
        best_disc_pts = None
        best_disc_beta = 0.0
        best_disc_center = None
        for axial in candidates:
            pts = axial_to_cartesian(axial)
            S, beta, center = optimize_shape(pts)
            if S < best_disc_S:
                best_disc_S = S
                best_disc_pts = pts
                best_disc_beta = beta
                best_disc_center = center

        best_global_S = best_disc_S
        best_global_pts = best_disc_pts
        best_global_beta = best_disc_beta
        best_global_center = best_disc_center

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)

    SCALE = 1.0 + 1e-12
    pts_scaled = []
    for x, y in best_global_pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))

    pts_scaled = np.array(pts_scaled)

    max_proj = 0
    for k in range(6):
        ang = best_global_beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (pts_scaled[:, 0] - cx) * nx + (pts_scaled[:, 1] - cy) * ny
        max_proj = max(max_proj, np.max(projs), -np.min(projs))

    D_scaled = max_proj
    S_scaled = (D_scaled + APOTHEM) / APOTHEM

    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)

    return inner, center, side, rotation
