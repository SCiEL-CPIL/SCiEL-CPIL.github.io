import math
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar, linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # --- Candidate point sets (axial coordinates) ---
    candidates = []

    # Hexagonal cluster: center + 6 neighbors + 4 more
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    candidates.append(axial_1)

    # Star: center + 6 neighbors + 4 opposite
    axial_2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_2)

    # 3x3 block + 2 extras
    axial_3 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ]
    candidates.append(axial_3)

    # 4x3 block + 1 extra
    axial_4 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_4)

    # 3x4 block + 1 extra
    axial_5 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_5)

    # Cross-like
    axial_7 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_7)

    # Center + 6 + 4 asymmetric
    axial_9 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (-1, 3), (1, -2)
    ]
    candidates.append(axial_9)

    # 3x3 + 2 top
    axial_10 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_10)

    # 2+4+4+1
    axial_11 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3)
    ]
    candidates.append(axial_11)

    # Center + 6 + 4 (variant)
    axial_13 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, 2)
    ]
    candidates.append(axial_13)

    # Center + 6 + 4 (variant)
    axial_14 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, -1)
    ]
    candidates.append(axial_14)

    # 3x3 with extras
    axial_15 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1), (0, -1)
    ]
    candidates.append(axial_15)

    # Staircase
    axial_16 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 2)
    ]
    candidates.append(axial_16)

    # Center + 6 + 4 (variant)
    axial_17 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (1, 2), (-1, 2), (0, 3)
    ]
    candidates.append(axial_17)

    # 4+3+3
    axial_18 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_18)

    # 2+3+2+3+1
    axial_19 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 3)
    ]
    candidates.append(axial_19)

    # Center + 6 + 4 (variant)
    axial_20 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, 1)
    ]
    candidates.append(axial_20)

    # 4+4+3
    axial_21 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_21)

    # 3+4+3+1
    axial_22 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3)
    ]
    candidates.append(axial_22)

    # Center + 6 + 4 (variant)
    axial_23 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_23)

    # 3+2+3+3
    axial_25 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_25)

    # 4+3+4+2
    axial_26 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_26)

    # Center + 6 + 4 (variant)
    axial_27 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, -2)
    ]
    candidates.append(axial_27)

    # More diverse configurations
    axial_29 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2), (4, 2)
    ]
    candidates.append(axial_29)

    axial_30 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 1), (-2, 1), (0, 2), (1, -2)
    ]
    candidates.append(axial_30)

    axial_31 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_31)

    axial_32 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2), (4, 2)
    ]
    candidates.append(axial_32)

    axial_33 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
        (0, 2), (1, 2)
    ]
    candidates.append(axial_33)

    axial_34 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 2), (-1, 2)
    ]
    candidates.append(axial_34)

    axial_35 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_35)

    # --- Geometry helpers ---

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def get_normals(beta):
        """Return 6 unit normals of the containing hexagon for rotation beta."""
        return np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])

    def compute_D(pts, center, normals):
        """Exact: D = max over points and normals of |(p - c) . n|."""
        centered = pts - np.array(center)
        projs = centered @ normals.T  # (n_pts, 6)
        return float(np.max(np.abs(projs)))

    def calc_S_for_beta(pts, beta):
        """
        Calculate minimal side length S for a fixed rotation beta.
        Uses exact LP to find optimal center minimizing D.
        """
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None

        normals = get_normals(beta)
        n_pts = len(pts)
        # Vectorized LP setup
        # For each point i and normal k:
        #   -nx*cx - ny*cy - t <= -proj  and  nx*cx + ny*cy - t <= proj
        projs = pts @ normals.T  # (n_pts, 6)
        # Build A and b in vectorized form
        # A has 2*n_pts*6 rows, 3 columns
        # Row pattern for each (i,k): [-nx, -ny, -1] and [nx, ny, -1]
        nx_arr = normals[:, 0]  # (6,)
        ny_arr = normals[:, 1]  # (6,)
        # For each point i, normal k: proj = projs[i, k]
        # A rows: for each i, for each k: [-nx_k, -ny_k, -1] and [nx_k, ny_k, -1]
        # b rows: -proj[i,k] and proj[i,k]
        # Total rows: n_pts * 6 * 2
        A = np.zeros((n_pts * 12, 3))
        b = np.zeros(n_pts * 12)
        # For each point i
        for i in range(n_pts):
            base = i * 12
            for k in range(6):
                r1 = base + k * 2
                r2 = r1 + 1
                A[r1, 0] = -nx_arr[k]
                A[r1, 1] = -ny_arr[k]
                A[r1, 2] = -1.0
                b[r1] = -projs[i, k]
                A[r2, 0] = nx_arr[k]
                A[r2, 1] = ny_arr[k]
                A[r2, 2] = -1.0
                b[r2] = projs[i, k]
        c_obj = np.array([0.0, 0.0, 1.0])
        try:
            res = linprog(c_obj, A_ub=A, b_ub=b, bounds=[(None, None), (None, None), (0, None)])
            if res.status == 0:
                D_opt = float(res.fun)
                c_opt = (float(res.x[0]), float(res.x[1]))
            else:
                c0 = np.mean(pts, axis=0)
                centered = pts - c0
                projs_c = centered @ normals.T
                D_opt = float(np.max(np.abs(projs_c)))
                c_opt = (float(c0[0]), float(c0[1]))
        except Exception:
            c0 = np.mean(pts, axis=0)
            centered = pts - c0
            projs_c = centered @ normals.T
            D_opt = float(np.max(np.abs(projs_c)))
            c_opt = (float(c0[0]), float(c0[1]))

        S = (D_opt + APOTHEM) / APOTHEM
        return S, c_opt

    def smooth_D(pts, center, normals, temperature):
        """Smooth surrogate for D via log-sum-exp soft-max over |projections|."""
        centered = pts - np.array(center)
        projs = centered @ normals.T  # (n_pts, 6)
        all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])
        max_val = float(np.max(all_projs))
        shifted = (all_projs - max_val) / temperature
        shifted = np.clip(shifted, -60.0, 60.0)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
        return temperature * log_sum

    def smooth_calc_S(pts, beta, center, temperature):
        normals = get_normals(beta)
        D = smooth_D(pts, center, normals, temperature)
        S = (D + APOTHEM) / APOTHEM
        return S

    def epigraph_objective(pts, params, temperature):
        """
        Epigraph formulation: minimize t + penalty
        s.t. smooth constraints: smooth_D <= t
        Uses a smooth penalty for the constraint.
        """
        cx, cy, t = params
        beta = ((params[3] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6) if len(params) > 3 else 0.0
        normals = get_normals(beta)
        D_smooth = smooth_D(pts, (cx, cy), normals, temperature)
        # Penalty for D_smooth > t
        violation = max(0.0, D_smooth - t)
        penalty = 100.0 * violation + 10000.0 * violation * violation
        return t + penalty

    def exact_objective_3d(pts, params):
        """Exact objective over (cx, cy, beta) — for final polish."""
        cx, cy, beta = params
        beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
        normals = get_normals(beta)
        D = compute_D(pts, (cx, cy), normals)
        return (D + APOTHEM) / APOTHEM

    def optimize_shape(pts):
        """Optimize (center, rotation) for a fixed point set.
        Strategy:
          1. Coarse grid search over beta with exact LP center solve.
          2. Bounded scalar refine around best beta.
          3. Smooth epigraph Nelder-Mead in (cx, cy, t, beta) with annealing.
          4. Exact high-precision polish from multiple starts.
        """
        best_S = float('inf')
        best_beta = 0.0
        best_center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))

        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, best_center

        # --- Phase 1: Coarse grid over beta (exact LP per beta) ---
        n_coarse = 121
        for i in range(n_coarse):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            beta = i * (math.pi / 6) / (n_coarse - 1)
            S, center = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                if center is not None:
                    best_center = center

        # --- Phase 2: Bounded scalar refine around best beta (multiple windows) ---
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                windows = [
                    (max(0.0, best_beta - math.pi / 30), min(math.pi / 6, best_beta + math.pi / 30)),
                    (max(0.0, best_beta - math.pi / 60), min(math.pi / 6, best_beta + math.pi / 60)),
                    (max(0.0, best_beta - math.pi / 120), min(math.pi / 6, best_beta + math.pi / 120)),
                    (max(0.0, best_beta - math.pi / 240), min(math.pi / 6, best_beta + math.pi / 240)),
                ]
                for lo, hi in windows:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    if hi - lo < 1e-15:
                        continue
                    try:
                        res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0],
                                              bounds=(lo, hi), method='bounded',
                                              options={'maxiter': 800, 'xatol': 1e-15})
                        if res.fun < best_S:
                            best_S = float(res.fun)
                            best_beta = float(res.x)
                            _, best_center = calc_S_for_beta(pts, best_beta)
                    except Exception:
                        pass
        except Exception:
            pass

        # --- Phase 3: Smooth epigraph Nelder-Mead with annealing ---
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                temperatures = [0.15, 0.08, 0.04, 0.02, 0.01, 0.005, 0.002]
                start_betas = [best_beta, best_beta + 0.02, best_beta - 0.02,
                               0.0, math.pi / 12, math.pi / 6]

                for temp in temperatures:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    for sb in start_betas:
                        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                            break

                        def smooth_obj_4d(params, t=temp):
                            cx, cy, t_val, beta = params
                            beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                            normals = get_normals(beta)
                            D_smooth = smooth_D(pts, (cx, cy), normals, t)
                            violation = max(0.0, D_smooth - t_val)
                            penalty = 50.0 * violation + 5000.0 * violation * violation
                            return t_val + penalty

                        init = [best_center[0], best_center[1], best_S * APOTHEM - APOTHEM, sb]
                        try:
                            res = minimize(smooth_obj_4d, init, method='Nelder-Mead',
                                           options={'maxiter': 600, 'xatol': 1e-13, 'fatol': 1e-13})
                            if res.fun < 1e10:
                                cx_r, cy_r, t_r, beta_r = res.x
                                beta_r = ((beta_r % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                                S_exact, c_exact = calc_S_for_beta(pts, beta_r)
                                if S_exact < best_S:
                                    best_S = S_exact
                                    best_beta = beta_r
                                    if c_exact is not None:
                                        best_center = c_exact
                        except Exception:
                            pass
        except Exception:
            pass

        # --- Phase 4: High-precision polish ---
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                cx0, cy0 = best_center

                # 4a: Very fine scalar search on beta with LP center solve
                for window in [math.pi / 400, math.pi / 800, math.pi / 1600, math.pi / 3200]:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    lo = max(0.0, best_beta - window)
                    hi = min(math.pi / 6, best_beta + window)
                    if hi - lo < 1e-15:
                        continue
                    try:
                        res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0],
                                              bounds=(lo, hi), method='bounded',
                                              options={'maxiter': 1200, 'xatol': 1e-16})
                        if res.fun < best_S:
                            best_S = float(res.fun)
                            best_beta = float(res.x)
                            _, best_center = calc_S_for_beta(pts, best_beta)
                    except Exception:
                        pass

                # 4b: Nelder-Mead + SLSQP polish in 3D
                polish_starts = [
                    [cx0, cy0, best_beta],
                    [cx0, cy0, best_beta + 0.003],
                    [cx0, cy0, best_beta - 0.003],
                    [cx0 + 0.003, cy0, best_beta],
                    [cx0, cy0 + 0.003, best_beta],
                    [cx0 - 0.003, cy0, best_beta],
                    [cx0, cy0 - 0.003, best_beta],
                    [cx0 + 0.002, cy0 + 0.002, best_beta],
                    [cx0 - 0.002, cy0 - 0.002, best_beta],
                ]

                for init in polish_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    try:
                        res = minimize(exact_objective_3d, init, method='Nelder-Mead',
                                       options={'maxiter': 4000, 'xatol': 1e-15, 'fatol': 1e-15})
                        if res.fun < best_S:
                            best_S = float(res.fun)
                            best_center = (float(res.x[0]), float(res.x[1]))
                            best_beta = ((float(res.x[2]) % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    except Exception:
                        pass

                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        continue
                    try:
                        res2 = minimize(exact_objective_3d,
                                        [best_center[0], best_center[1], best_beta],
                                        method='SLSQP',
                                        options={'maxiter': 2000, 'ftol': 1e-18})
                        if res2.fun < best_S:
                            best_S = float(res2.fun)
                            best_center = (float(res2.x[0]), float(res2.x[1]))
                            best_beta = ((float(res2.x[2]) % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    except Exception:
                        pass

                # 4c: Final LP center re-solve at the best beta
                if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                    S_final, c_final = calc_S_for_beta(pts, best_beta)
                    if c_final is not None and S_final < best_S:
                        best_S = S_final
                        best_center = c_final

                # 4d: High-precision center polish using active set refinement
                if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                    # After LP, identify active constraints and refine
                    cx0, cy0 = best_center
                    normals = get_normals(best_beta)
                    # Compute projections at current center
                    centered = pts - np.array([cx0, cy0])
                    projs = centered @ normals.T  # (n, 6)
                    # Active constraints: |proj| close to D
                    D_cur = float(np.max(np.abs(projs)))
                    active_thresh = D_cur - 1e-10
                    # Find which (point, normal, sign) are active
                    active_pts = []
                    active_normals = []
                    active_sigs = []
                    for i in range(len(pts)):
                        for k in range(6):
                            if projs[i, k] > active_thresh:
                                active_pts.append(i)
                                active_normals.append(k)
                                active_sigs.append(1)
                            if -projs[i, k] > active_thresh:
                                active_pts.append(i)
                                active_normals.append(k)
                                active_sigs.append(-1)
                    if len(active_pts) >= 2:
                        # Solve least-squares for center using active constraints
                        # For each active constraint: sig * (pts[i].n[k] - c.n[k]) = D
                        # => c.n[k] = pts[i].n[k] - sig * D
                        # This is an overdetermined system; use least squares
                        A_act = np.array([normals[ak] for ak in active_normals])
                        b_act = np.array([pts[ai, 0] * normals[ak, 0] + pts[ai, 1] * normals[ak, 1] - sig * D_cur
                                          for ai, ak, sig in zip(active_pts, active_normals, active_sigs)])
                        try:
                            c_ls, _, _, _ = np.linalg.lstsq(A_act, b_act, rcond=None)
                            c_new = (float(c_ls[0]), float(c_ls[1]))
                            # Verify improvement
                            D_new = compute_D(pts, c_new, normals)
                            if D_new < D_cur - 1e-14:
                                best_center = c_new
                                best_S = (D_new + APOTHEM) / APOTHEM
                        except Exception:
                            pass
        except Exception:
            pass

        return best_S, best_beta, best_center

    # --- Main loop over candidates ---
    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    # Sort candidates by a quick heuristic: bounding box area in axial space
    def axial_bbox_area(axial_list):
        qs = [q for q, r in axial_list]
        rs = [r for r, c in axial_list]
        return (max(qs) - min(qs) + 1) * (max(rs) - min(rs) + 1)

    sorted_candidates = sorted(candidates, key=axial_bbox_area)

    # Pre-filter: compute a lower bound using bounding box diagonal
    # For a set of points, the minimum enclosing hexagon side length is at least
    # (max projection onto any axis + APOTHEM) / APOTHEM for some axis
    # Quick lower bound: use the minimum over 6 standard directions of the
    # range of projections, which gives a lower bound on D
    def quick_lower_bound(pts):
        """Quick lower bound on S using bounding box in 6 hex directions."""
        # The 6 unit normals for beta=0
        normals = get_normals(0.0)
        projs = pts @ normals.T  # (n, 6)
        # For each normal, the range is max - min
        ranges = np.max(projs, axis=0) - np.min(projs, axis=0)
        # D must be at least max(range_k) / 2 for some... actually D >= max_k (range_k)/2
        # But the center can be chosen to minimize max |proj - c.n|
        # The minimum over center of max |proj - c.n| for a fixed normal is range/2
        # But D is max over all normals, so D >= max_k (range_k / 2)
        D_lb = float(np.max(ranges) / 2.0)
        S_lb = (D_lb + APOTHEM) / APOTHEM
        return S_lb

    # Pre-compute lower bounds and filter
    candidate_info = []
    for idx, axial in enumerate(sorted_candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        lb = quick_lower_bound(pts)
        candidate_info.append((lb, idx, axial, pts))

    # Sort by lower bound
    candidate_info.sort(key=lambda x: x[0])

    for lb, idx, axial, pts in candidate_info:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        # Skip if lower bound is already worse than best found
        if lb >= best_global_S and best_global_S < float('inf'):
            continue
        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

    # --- Fallback ---
    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = axial_to_cartesian(candidates[0])
        best_global_center = (float(np.mean(best_global_pts[:, 0])),
                              float(np.mean(best_global_pts[:, 1])))
        best_global_beta = 0.0
        normals = get_normals(best_global_beta)
        max_proj = 0.0
        for nx, ny in normals:
            projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
            max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))
        best_global_S = (max_proj + APOTHEM) / APOTHEM

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)

    # --- Final: recompute exact S with the chosen (center, beta), add tiny scale for safety ---
    SCALE = 1.0 + 1e-12
    pts_scaled = []
    for x, y in best_global_pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))
    pts_scaled = np.array(pts_scaled)

    normals = get_normals(best_global_beta)
    max_proj = 0.0
    for nx, ny in normals:
        projs = (pts_scaled[:, 0] - cx) * nx + (pts_scaled[:, 1] - cy) * ny
        max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))

    D_scaled = max_proj
    S_scaled = (D_scaled + APOTHEM) / APOTHEM

    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)

    return inner, center, side, rotation
