import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # --- Geometry helpers ---
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def get_normals(beta):
        """Return 6 unit outward normals of the containing hexagon for rotation beta.
        For a regular hexagon with side S, the distance from center to each side is S*sqrt(3)/2.
        The normals point outward from center to each side."""
        # The 6 sides are at angles beta, beta+pi/3, ... from the x-axis
        # Actually, let's think about this more carefully.
        # A regular hexagon centered at origin with one flat side up:
        # The sides are at angles 0, pi/3, 2pi/3, pi, 4pi/3, 5pi/3 from center
        # The outward normals to these sides are at angles 0, pi/3, ...
        # Wait, I need to be consistent with the LP formulation.
        # For a hexagon with side length S, the inradius (apothem) is S*sqrt(3)/2.
        # The constraint is that for each of the 6 half-planes defining the hexagon,
        # the projection of each inner hexagon center onto the normal must be <= S*sqrt(3)/2.
        # The 6 outward normals for a hexagon with a flat top (one side parallel to x-axis at top)
        # are at angles: pi/6, pi/6+pi/3, pi/6+2pi/3, pi/6+pi, pi/6+4pi/3, pi/6+5pi/3
        # But actually, for a hexagon with a point at top, the normals would be different.
        # Let's use: the 6 normals are at angles beta + k*pi/3 for k=0..5
        # where beta is the rotation angle of the hexagon.
        # The apothem (distance from center to each side) = S * sqrt(3)/2
        return np.array([
            (math.cos(beta + k * math.pi / 3.0),
             math.sin(beta + k * math.pi / 3.0))
            for k in range(6)
        ])

    def compute_D(pts, center, normals):
        """D = max over points and normals of (p - c) . n, which is the max projection.
        This is the inradius of the minimal enclosing hexagon."""
        centered = pts - np.array(center)
        projs = centered @ normals.T  # (n_pts, 6)
        return float(np.max(projs))

    def calc_S_for_beta(pts, beta):
        """LP over center (cx, cy) for fixed beta. Returns (S, center, D)."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, float('inf')

        normals = get_normals(beta)
        c_obj = np.array([0.0, 0.0, 1.0])
        
        n_pts = len(pts)
        # For each normal k (6 normals) and each point i (n_pts):
        # Constraint: (px_i - cx)*nx_k + (py_i - cy)*ny_k <= D
        # => -nx_k*cx - ny_k*cy - D <= -px_i*nx_k - py_i*ny_k
        # That's 6 * n_pts constraints.
        n_constraints = 6 * n_pts
        A_ub = np.zeros((n_constraints, 3))
        b_ub = np.zeros(n_constraints)
        
        for k in range(6):
            nx, ny = normals[k]
            idx = k * n_pts
            A_ub[idx:idx+n_pts, 0] = -nx
            A_ub[idx:idx+n_pts, 1] = -ny
            A_ub[idx:idx+n_pts, 2] = -1.0
            b_ub[idx:idx+n_pts] = -(pts[:, 0] * nx + pts[:, 1] * ny)

        bounds = [(None, None), (None, None), (0, None)]

        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = float(res.x[2])
                cx, cy = float(res.x[0]), float(res.x[1])
                S = D / APOTHEM
                return S, (cx, cy), D
            else:
                return float('inf'), None, float('inf')
        except Exception:
            return float('inf'), None, float('inf')

    def smooth_D(pts, center, normals, temperature):
        """Smooth surrogate for D via log-sum-exp."""
        centered = pts - np.array(center)
        projs = centered @ normals.T
        all_projs = projs.ravel()
        max_val = float(np.max(all_projs))
        shifted = (all_projs - max_val) / temperature
        shifted = np.clip(shifted, -60.0, 60.0)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
        return temperature * log_sum

    def smooth_calc_S(pts, beta, center, temperature):
        normals = get_normals(beta)
        D = smooth_D(pts, center, normals, temperature)
        S = D / APOTHEM
        return S

    def exact_objective_3d(pts, params):
        """Exact objective over (cx, cy, beta)."""
        cx, cy, beta = params
        beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
        normals = get_normals(beta)
        D = compute_D(pts, (cx, cy), normals)
        return D / APOTHEM

    # --- Candidate point sets (axial coordinates) ---
    # Generate many diverse compact configurations for 11 hexagons
    candidates = []

    # Hexagonal cluster: center + 6 neighbors + 4 more
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    candidates.append(axial_1)

    # Compact 3x3 hexagonal arrangement
    axial_2 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_2)

    # Dense 4x4 block
    axial_3 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_3)

    # Star: center + 6 neighbors + 4 opposite
    axial_4 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_4)

    # 3x3 block + 2 extras
    axial_5 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ]
    candidates.append(axial_5)

    # 4x3 block + 1 extra
    axial_6 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_6)

    # 3x4 block + 1 extra
    axial_7 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_7)

    # 3x3 + 2
    axial_8 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_8)

    # Cross-like
    axial_9 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_9)

    # 5+6 row
    axial_10 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)
    ]
    candidates.append(axial_10)

    # Center + 6 + 4 asymmetric
    axial_11 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (-1, 3), (1, -2)
    ]
    candidates.append(axial_11)

    # 3x3 + 2 top
    axial_12 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_12)

    # Compact 3x3 with corners filled
    axial_13 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1), (1, 3)
    ]
    candidates.append(axial_13)

    # Very compact: 2-3-2-3-1
    axial_14 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3), (2, 3),
        (1, 4)
    ]
    candidates.append(axial_14)

    # 4-2-3-2
    axial_15 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_15)

    # Center + 6 + 4 (tight cluster)
    axial_16 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (1, 2), (-1, 3)
    ]
    candidates.append(axial_16)

    # 2+4+4+1
    axial_17 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3)
    ]
    candidates.append(axial_17)

    # 4+2+2+2+1
    axial_18 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_18)

    # Center + 6 + 4 (variant)
    axial_19 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, 2)
    ]
    candidates.append(axial_19)

    # Center + 6 + 4 (variant)
    axial_20 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, -1)
    ]
    candidates.append(axial_20)

    # 3x3 with extras
    axial_21 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1), (0, -1)
    ]
    candidates.append(axial_21)

    # Staircase
    axial_22 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 2)
    ]
    candidates.append(axial_22)

    # Center + 6 + 4 (variant)
    axial_23 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (1, 2), (-1, 2), (0, 3)
    ]
    candidates.append(axial_23)

    # 4+3+3
    axial_24 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_24)

    # 2+3+2+3+1
    axial_25 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 3)
    ]
    candidates.append(axial_25)

    # Center + 6 + 4 (variant)
    axial_26 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, 1)
    ]
    candidates.append(axial_26)

    # 4+4+3
    axial_27 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_27)

    # 3+4+3+1
    axial_28 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3)
    ]
    candidates.append(axial_28)

    # Center + 6 + 4 (variant)
    axial_29 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_29)

    # 2+2+3+2+2
    axial_30 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_30)

    # 3+2+3+3
    axial_31 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_31)

    # Very compact star
    axial_32 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, -1)
    ]
    candidates.append(axial_32)

    # Compact 2+3+2+3
    axial_33 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_33)

    # 4+3+4+2
    axial_34 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_34)

    # Center + 6 + 4 (variant)
    axial_35 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, -2)
    ]
    candidates.append(axial_35)

    # 2+3+2+3+3
    axial_36 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3), (2, 3),
        (1, 4), (2, 4)
    ]
    candidates.append(axial_36)

    # More diverse configurations
    axial_37 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2), (4, 2)
    ]
    candidates.append(axial_37)

    axial_38 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 1), (-2, 1), (0, 2), (1, -2)
    ]
    candidates.append(axial_38)

    axial_39 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_39)

    axial_40 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2), (4, 2)
    ]
    candidates.append(axial_40)

    axial_41 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
        (0, 2), (1, 2)
    ]
    candidates.append(axial_41)

    axial_42 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 2), (-1, 2)
    ]
    candidates.append(axial_42)

    axial_43 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_43)

    # New diverse configurations
    axial_44 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 0), (3, 0)
    ]
    candidates.append(axial_44)

    axial_45 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2), (3, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_45)

    axial_46 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3), (3, 3)
    ]
    candidates.append(axial_46)

    axial_47 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_47)

    axial_48 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3),
        (1, 4)
    ]
    candidates.append(axial_48)

    axial_49 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_49)

    axial_50 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_50)

    def optimize_shape(pts):
        """Optimize (center, rotation) for a fixed point set."""
        best_S = float('inf')
        best_beta = 0.0
        best_center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))

        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, best_center

        # --- Phase 1: Coarse grid over beta ---
        n_coarse = 121
        for i in range(n_coarse):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            beta = i * (math.pi / 6) / (n_coarse - 1)
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                if center is not None:
                    best_center = center

        # --- Phase 2: Bounded scalar refine ---
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                lo = max(0.0, best_beta - math.pi / 40)
                hi = min(math.pi / 6, best_beta + math.pi / 40)
                res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0],
                                      bounds=(lo, hi), method='bounded',
                                      options={'maxiter': 400, 'xatol': 1e-14})
                if res.fun < best_S:
                    best_S = float(res.fun)
                    best_beta = float(res.x)
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
                
                res_global = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0],
                                             bounds=(0.0, math.pi / 6), method='bounded',
                                             options={'maxiter': 600, 'xatol': 1e-13})
                if res_global.fun < best_S:
                    best_S = float(res_global.fun)
                    best_beta = float(res_global.x)
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass

        # --- Phase 3: Smooth-surrogate Nelder-Mead with annealing ---
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                temperatures = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
                start_betas = [best_beta, best_beta + 0.02, best_beta - 0.02,
                               best_beta + 0.005, best_beta - 0.005,
                               0.0, math.pi / 12, math.pi / 6]

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
                                           options={'maxiter': 1500, 'xatol': 1e-14, 'fatol': 1e-14})
                            if res.fun < 1e10:
                                cx_r, cy_r, beta_r = res.x
                                beta_r = ((beta_r % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                                S_exact, c_exact, _ = calc_S_for_beta(pts, beta_r)
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
                    [cx0, cy0, best_beta + 0.003],
                    [cx0, cy0, best_beta - 0.003],
                    [cx0, cy0, best_beta + 0.001],
                    [cx0, cy0, best_beta - 0.001],
                    [cx0 + 0.003, cy0, best_beta],
                    [cx0 - 0.003, cy0, best_beta],
                    [cx0, cy0 + 0.003, best_beta],
                    [cx0, cy0 - 0.003, best_beta],
                ]

                for init in polish_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    try:
                        res = minimize(exact_objective_3d, init, method='Nelder-Mead',
                                       options={'maxiter': 20000, 'xatol': 1e-16, 'fatol': 1e-16})
                        if res.fun < best_S:
                            best_S = float(res.fun)
                            best_center = (float(res.x[0]), float(res.x[1]))
                            best_beta = ((float(res.x[2]) % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    except Exception:
                        pass
        except Exception:
            pass

        # --- Phase 5: Final LP center polish at optimal beta ---
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                S_lp, c_lp, D_lp = calc_S_for_beta(pts, best_beta)
                if S_lp < best_S:
                    best_S = S_lp
                    if c_lp is not None:
                        best_center = c_lp
        except Exception:
            pass

        # --- Phase 6: Multi-start final polish at nearby betas ---
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                for db in [-0.002, -0.001, 0.001, 0.002]:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    b_try = best_beta + db
                    S_try, c_try, _ = calc_S_for_beta(pts, b_try)
                    if S_try < best_S - 1e-15:
                        best_S = S_try
                        best_beta = b_try
                        if c_try is not None:
                            best_center = c_try
        except Exception:
            pass

        return best_S, best_beta, best_center

    # --- Main loop over candidates ---
    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    # Score candidates by bounding box area and keep top ones
    scored_candidates = []
    for axial in candidates:
        pts = axial_to_cartesian(axial)
        xs, ys = pts[:, 0], pts[:, 1]
        area = (np.max(xs) - np.min(xs)) * (np.max(ys) - np.min(ys))
        scored_candidates.append((area, axial, pts))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    # Keep top 25 candidates for deeper optimization
    MAX_CANDIDATES = 25
    top_candidates = [x[1] for x in scored_candidates[:MAX_CANDIDATES]]
    top_pts = [x[2] for x in scored_candidates[:MAX_CANDIDATES]]

    for idx in range(len(top_candidates)):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = top_pts[idx]
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
        D = compute_D(best_global_pts, best_global_center, normals)
        best_global_S = D / APOTHEM

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)

    # --- Final high-precision polish: try a few more LP solves at nearby betas ---
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        for dbeta in [-0.003, -0.001, 0.0, 0.001, 0.003]:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            b_try = best_global_beta + dbeta
            S_try, c_try, _ = calc_S_for_beta(best_global_pts, b_try)
            if S_try < best_global_S - 1e-15:
                best_global_S = S_try
                best_global_beta = b_try
                if c_try is not None:
                    best_global_center = c_try
                    cx, cy = c_try
                    angle_deg = math.degrees(b_try)

    # --- Final: recompute exact S with the chosen (center, beta) ---
    normals = get_normals(best_global_beta)
    D_final = compute_D(best_global_pts, best_global_center, normals)
    S_final = D_final / APOTHEM

    # Add tiny safety margin to side length
    SCALE = 1.0 + 1e-10
    side = float(S_final * SCALE)

    inner = []
    for x, y in best_global_pts:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    rotation = float(angle_deg)

    return inner, center, side, rotation
