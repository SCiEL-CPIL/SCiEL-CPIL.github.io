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

    # Keep only the most diverse and promising candidates
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

    # Staircase
    axial_16 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 2)
    ]
    candidates.append(axial_16)

    # 4+3+3
    axial_18 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_18)

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

    # 4+3+4+2
    axial_26 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_26)

    # More diverse: 5+3+3
    axial_29 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2), (4, 2)
    ]
    candidates.append(axial_29)

    # 3+4+3+1 variant
    axial_31 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_31)

    # 2+5+4
    axial_32 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2), (4, 2)
    ]
    candidates.append(axial_32)

    # 5+5+1
    axial_33 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
        (0, 2), (1, 2)
    ]
    candidates.append(axial_33)

    # 3+2+3+3
    axial_35 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_35)

    # Additional diverse candidates
    # Compact 3-4-3 with offset
    axial_36 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3)
    ]
    candidates.append(axial_36)

    # Diamond-ish
    axial_37 = [
        (0, 0),
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, 1), (-1, 1),
        (2, 0), (-2, 0),
        (0, 2), (0, -2),
        (1, -1)
    ]
    candidates.append(axial_37)

    # 2+3+3+3
    axial_38 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3), (3, 3)
    ]
    candidates.append(axial_38)

    # 4+2+4+1
    axial_39 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (2, 3)
    ]
    candidates.append(axial_39)

    # 3+3+3+2
    axial_40 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_40)

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
        Uses LP for exact center optimization with vectorized setup.
        """
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None

        normals = get_normals(beta)
        c0 = np.mean(pts, axis=0)
        n_pts = len(pts)

        # Vectorized LP setup:
        # For each point i and normal j:
        #   (p_i - c) · n_j <= t  =>  -n_j·c - t <= -p_i·n_j
        #   -(p_i - c) · n_j <= t =>   n_j·c - t <=  p_i·n_j
        #
        # A_ub has shape (2*n_pts*6, 3), b_ub has shape (2*n_pts*6,)
        #
        # Build using broadcasting:
        # pts: (n_pts, 2), normals: (6, 2)
        # projections p_i · n_j: (n_pts, 6)
        proj = pts @ normals.T  # (n_pts, 6)

        # For constraint (p_i - c)·n_j <= t:
        #   -nx*cx - ny*cy - t <= -proj[i,j]
        # Row: [-nx, -ny, -1], rhs: -proj[i,j]
        #
        # For constraint -(p_i - c)·n_j <= t:
        #   nx*cx + ny*cy - t <= proj[i,j]
        # Row: [nx, ny, -1], rhs: proj[i,j]

        # Reshape for vectorized construction
        # normals: (6, 2) -> expand to (1, 6, 2)
        n_expanded = normals[np.newaxis, :, :]  # (1, 6, 2)
        proj_expanded = proj[:, :, np.newaxis]  # (n_pts, 6, 1)

        # Constraint 1: -n_j·c - t <= -proj
        # A rows: [-nx, -ny, -1] for each (i, j)
        A1 = np.zeros((n_pts, 6, 3))
        A1[:, :, 0] = -normals[:, 0][np.newaxis, :]  # broadcast over pts
        A1[:, :, 1] = -normals[:, 1][np.newaxis, :]
        A1[:, :, 2] = -1.0
        b1 = -proj  # (n_pts, 6)

        # Constraint 2: n_j·c - t <= proj
        A2 = np.zeros((n_pts, 6, 3))
        A2[:, :, 0] = normals[:, 0][np.newaxis, :]
        A2[:, :, 1] = normals[:, 1][np.newaxis, :]
        A2[:, :, 2] = -1.0
        b2 = proj  # (n_pts, 6)

        A_ub = np.concatenate([A1.reshape(-1, 3), A2.reshape(-1, 3)], axis=0)
        b_ub = np.concatenate([b1.ravel(), b2.ravel()])

        c_obj = np.array([0.0, 0.0, 1.0])

        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub,
                          bounds=[(None, None), (None, None), (0, None)],
                          method='highs')
            if res.status == 0:
                cx_opt, cy_opt, t_opt = res.x
                D_opt = t_opt
                S = (D_opt + APOTHEM) / APOTHEM
                return S, (float(cx_opt), float(cy_opt))
        except Exception:
            pass

        # Fallback to Nelder-Mead
        def D_of_c(c):
            centered = pts - c
            projs = centered @ normals.T
            return float(np.max(np.abs(projs)))

        try:
            res = minimize(D_of_c, c0, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-12})
            c_opt = res.x
            D_opt = res.fun
        except Exception:
            c_opt = c0
            D_opt = D_of_c(c0)

        S = (D_opt + APOTHEM) / APOTHEM
        return S, (float(c_opt[0]), float(c_opt[1]))

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
          1. Coarse grid search over beta with fast center solve.
          2. Bounded scalar refine around best beta.
          3. Smooth-surrogate Nelder-Mead in (cx, cy, beta) with annealing.
          4. Exact high-precision polish from multiple starts.
        """
        best_S = float('inf')
        best_beta = 0.0
        best_center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))

        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, best_center

        # --- Phase 1: Coarse grid over beta ---
        n_coarse = 61
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

        # --- Phase 2: Bounded scalar refine around best beta ---
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                lo = max(0.0, best_beta - math.pi / 90)
                hi = min(math.pi / 6, best_beta + math.pi / 90)
                res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0],
                                      bounds=(lo, hi), method='bounded',
                                      options={'maxiter': 400, 'xatol': 1e-14})
                if res.fun < best_S:
                    best_S = float(res.fun)
                    best_beta = float(res.x)
                    _, best_center = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass

        # --- Phase 3: Smooth-surrogate Nelder-Mead with annealing ---
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                temperatures = [0.1, 0.05, 0.02]
                start_betas = [best_beta, best_beta + 0.03, best_beta - 0.03]

                for temp in temperatures:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    for sb in start_betas:
                        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                            break

                        def smooth_obj(params, t=temp):
                            cx, cy, beta = params
                            beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                            return smooth_calc_S(pts, beta, (cx, cy), t)

                        init = [best_center[0], best_center[1], sb]
                        try:
                            res = minimize(smooth_obj, init, method='Nelder-Mead',
                                           options={'maxiter': 400, 'xatol': 1e-12, 'fatol': 1e-12})
                            if res.fun < 1e10:
                                cx_r, cy_r, beta_r = res.x
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

        # --- Phase 4: Exact high-precision polish ---
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                cx0, cy0 = best_center
                polish_starts = [
                    [cx0, cy0, best_beta],
                    [cx0, cy0, best_beta + 0.01],
                    [cx0, cy0, best_beta - 0.01],
                    [cx0 + 0.01, cy0, best_beta],
                    [cx0, cy0 + 0.01, best_beta],
                    [cx0 - 0.01, cy0, best_beta],
                    [cx0, cy0 - 0.01, best_beta],
                    [cx0 + 0.02, cy0 + 0.01, best_beta],
                    [cx0 - 0.02, cy0 - 0.01, best_beta],
                ]

                for init in polish_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    # Nelder-Mead polish
                    try:
                        res = minimize(exact_objective_3d, init, method='Nelder-Mead',
                                       options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14})
                        if res.fun < best_S:
                            best_S = float(res.fun)
                            best_center = (float(res.x[0]), float(res.x[1]))
                            best_beta = ((float(res.x[2]) % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    except Exception:
                        pass

                    # SLSQP polish from current best
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        continue
                    try:
                        res2 = minimize(exact_objective_3d,
                                        [best_center[0], best_center[1], best_beta],
                                        method='SLSQP',
                                        options={'maxiter': 500, 'ftol': 1e-15})
                        if res2.fun < best_S:
                            best_S = float(res2.fun)
                            best_center = (float(res2.x[0]), float(res2.x[1]))
                            best_beta = ((float(res2.x[2]) % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
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

    for idx, axial in enumerate(candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
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
