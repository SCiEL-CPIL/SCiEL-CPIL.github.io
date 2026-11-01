import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar, differential_evolution

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 145.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Candidate axial coordinate sets for 11 hexagons
    candidates = []

    # Flower + 4 from next ring (original)
    candidates.append([
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ])

    # Flower + 4 cardinal from next ring
    candidates.append([
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ])

    # 3x3 block + 2 extra
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ])

    # 4x3 block minus 1
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ])

    # 3x4 block minus 1
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ])

    # 3x3 + 2 diagonal
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ])

    # Star of 7 + 4 outer
    candidates.append([
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ])

    # 5+6 row
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)
    ])

    # Flower + 4 mixed next ring
    candidates.append([
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (-1, 3), (1, -2)
    ])

    # 3x3 + 2 top-right
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ])

    # 2+3+3+3 staggered
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3)
    ])

    # Vertical strip
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4)
    ])

    # Flower + 2 opposite + 2 adjacent
    candidates.append([
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, -1)
    ])

    # 3x3 + 3 extras
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1), (0, -1)
    ])

    # 2+3+2+3+1
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 2)
    ])

    # Flower + 4 ring-2
    candidates.append([
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (1, 2), (-1, 2), (0, 3)
    ])

    # 4+3+3 trapezoid
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2), (3, 2)
    ])

    # 2+3+3+3
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 3)
    ])

    # 4+3+2+2
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2)
    ])

    # 3+4+3+1
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3)
    ])

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def solve_lp_for_beta(pts, beta):
        """Solve LP for given points and outer hexagon rotation. Returns (S, cx, cy) or (inf, None, None)."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None

        normals = np.array([
            [math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)]
            for k in range(6)
        ])

        c = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []

        for nx, ny in normals:
            for px, py in pts:
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px * nx + py * ny))
                A_ub.append([nx, ny, -1.0])
                b_ub.append(px * nx + py * ny)

        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(None, None), (None, None), (0, None)]

        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = res.x[2]
                S = (D + APOTHEM) / APOTHEM
                return S, res.x[0], res.x[1]
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    def optimize_rotation(pts):
        """Find optimal rotation for a fixed point set. Returns (best_S, best_beta, best_cx, best_cy)."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None, None

        best_S = float('inf')
        best_beta = 0.0
        best_cx = 0.0
        best_cy = 0.0

        # Coarse scan
        n_scan = 37
        for i in range(n_scan):
            beta = i * (math.pi / 6.0) / (n_scan - 1)
            S, cx, cy = solve_lp_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_cx = cx
                best_cy = cy

        if best_S == float('inf'):
            return float('inf'), 0.0, None, None

        # Fine scan around best
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            step = 0.01
            for i in range(-50, 51):
                beta = best_beta + i * step
                if beta < -1e-6 or beta > math.pi / 6 + 1e-6:
                    continue
                beta = max(0.0, min(math.pi / 6, beta))
                S, cx, cy = solve_lp_for_beta(pts, beta)
                if S < best_S:
                    best_S = S
                    best_beta = beta
                    best_cx = cx
                    best_cy = cy

        # minimize_scalar polish
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            def obj(beta):
                S, _, _ = solve_lp_for_beta(pts, beta)
                return S

            try:
                local_lo = max(0.0, best_beta - 0.02)
                local_hi = min(math.pi / 6, best_beta + 0.02)
                res = minimize_scalar(obj, bounds=(local_lo, local_hi), method='bounded',
                                      options={'maxiter': 200, 'xatol': 1e-10})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_cx, best_cy = solve_lp_for_beta(pts, best_beta)
            except Exception:
                pass

        return best_S, best_beta, best_cx, best_cy

    # Phase 1: Evaluate all candidate topologies
    best_overall_S = float('inf')
    best_overall_pts = None
    best_overall_beta = 0.0
    best_overall_cx = 0.0
    best_overall_cy = 0.0

    for axial in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        S, beta, cx, cy = optimize_rotation(pts)
        if S < best_overall_S:
            best_overall_S = S
            best_overall_pts = pts
            best_overall_beta = beta
            best_overall_cx = cx
            best_overall_cy = cy

    if best_overall_S == float('inf'):
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Phase 2: Joint continuous optimization of point positions + rotation
    # Variables: 11*(x,y) + beta = 23 variables
    # Objective: minimize S (outer hexagon side length)
    # Constraints: pairwise distance >= 2 (hexagons of side 1, touching at edges)

    n_pts = 11

    def compute_S_from_positions(pos_flat, beta):
        """Given 22 positions (flat) and beta, compute the enclosing hexagon side length."""
        pts = pos_flat.reshape(n_pts, 2)
        normals = np.array([
            [math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)]
            for k in range(6)
        ])

        # For each normal, compute max and min projection of all points
        # The enclosing hexagon must contain all points
        # For normal k: max_i (p_i . n_k) - min_i (p_i . n_k) <= 2 * D
        # where D is the half-width in that direction
        # The hexagon side S relates to D: D = S * APOTHEM
        # Actually for a regular hexagon with side S, the apothem is S*sqrt(3)/2
        # The projection width along a normal is 2*S*sqrt(3)/2 = S*sqrt(3) for the direction
        # Wait, let me think more carefully.

        # A regular hexagon with side length S has:
        # - Apothem (distance from center to side) = S * sqrt(3)/2
        # - The hexagon is the intersection of 6 half-planes: |p . n_k| <= S * sqrt(3)/2
        # where n_k are the 6 outward normals (pointing to the midpoints of sides)

        # So for our points to fit: for all k, max_i (p_i - c).n_k <= S*APOTHEM
        # and min_i (p_i - c).n_k >= -S*APOTHEM
        # i.e., max_i (p_i - c).n_k - min_i (p_i - c).n_k <= 2 * S * APOTHEM

        # To minimize S, we choose c to minimize the max over k of (max_i (p_i-c).n_k - min_i (p_i-c).n_k) / 2
        # This is the minimum enclosing hexagon problem.

        # For a fixed beta, this is an LP in (cx, cy, D):
        # minimize D
        # s.t. (p_i - c).n_k <= D for all i, k
        #      -(p_i - c).n_k <= D for all i, k

        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []

        for nx, ny in normals:
            for px, py in pts:
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px * nx + py * ny))
                A_ub.append([nx, ny, -1.0])
                b_ub.append(px * nx + py * ny)

        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(None, None), (None, None), (0, None)]

        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = res.x[2]
                S = (D + APOTHEM) / APOTHEM
                return S
            else:
                return 1e6
        except Exception:
            return 1e6

    def smooth_objective(x):
        """Smooth objective for joint optimization. x = [x1,y1,...,x11,y11, beta]"""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return 1e10

        pos_flat = x[:2 * n_pts]
        beta = x[2 * n_pts]
        pts = pos_flat.reshape(n_pts, 2)

        # Compute S using smooth approximation
        normals = np.array([
            [math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)]
            for k in range(6)
        ])

        # For each normal, compute the width (max - min projection)
        # We need to center optimally, so use the midpoint
        # width_k = max_i (p_i . n_k) - min_i (p_i . n_k)
        # S = max_k (width_k / (2 * APOTHEM)) + 1  (since D = S*APOTHEM - APOTHEM, width = 2*D)
        # Actually: width_k = 2 * D_k where D_k = max_i (p_i-c).n_k (with optimal c)
        # S = D / APOTHEM where D = max_k D_k
        # But D_k = (max_i (p_i . n_k) - min_i (p_i . n_k)) / 2

        widths = np.zeros(6)
        for k in range(6):
            projs = pts @ normals[k]
            widths[k] = (np.max(projs) - np.min(projs)) / 2.0

        # Smooth max of widths / APOTHEM
        tau = 0.05
        vals = widths / APOTHEM
        max_val = np.max(vals)
        if max_val > 50:
            lse = max_val + tau * np.log(np.sum(np.exp((vals - max_val) / tau)))
        else:
            lse = tau * np.log(np.sum(np.exp(vals / tau)))
        S_smooth = lse

        # Pairwise distance penalty (soft)
        # For each pair, distance must be >= 2
        # penalty = sum over pairs of max(0, 2 - dist_i)^2
        penalty = 0.0
        for i in range(n_pts):
            for j in range(i + 1, n_pts):
                dx = pts[i, 0] - pts[j, 0]
                dy = pts[i, 1] - pts[j, 1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < 2.0:
                    penalty += (2.0 - dist) ** 2

        return S_smooth + 100.0 * penalty

    # Initialize from best candidate
    x0 = np.zeros(2 * n_pts + 1)
    x0[:2 * n_pts] = best_overall_pts.flatten()
    x0[2 * n_pts] = best_overall_beta

    best_joint_S = best_overall_S
    best_joint_x = x0.copy()

    # Try L-BFGS-B
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        try:
            bounds_joint = []
            for _ in range(2 * n_pts):
                bounds_joint.append((-5.0, 5.0))
            bounds_joint.append((0.0, math.pi / 6))

            res = minimize(smooth_objective, x0, method='L-BFGS-B', bounds=bounds_joint,
                           options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6})

            if res.fun < best_joint_S:
                # Verify with exact LP
                pos_flat = res.x[:2 * n_pts]
                beta = res.x[2 * n_pts]
                S_exact = compute_S_from_positions(pos_flat, beta)

                # Check pairwise distances
                pts = pos_flat.reshape(n_pts, 2)
                min_dist = float('inf')
                for i in range(n_pts):
                    for j in range(i + 1, n_pts):
                        dx = pts[i, 0] - pts[j, 0]
                        dy = pts[i, 1] - pts[j, 1]
                        d = math.sqrt(dx * dx + dy * dy)
                        min_dist = min(min_dist, d)

                if min_dist >= 2.0 - 1e-6 and S_exact < best_joint_S:
                    best_joint_S = S_exact
                    best_joint_x = res.x.copy()
        except Exception:
            pass

    # Phase 3: Differential Evolution for global search on joint variables
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        try:
            def de_objective(x):
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    return 1e10
                pos_flat = x[:2 * n_pts]
                beta = x[2 * n_pts]
                pts = pos_flat.reshape(n_pts, 2)

                # Check pairwise distances
                for i in range(n_pts):
                    for j in range(i + 1, n_pts):
                        dx = pts[i, 0] - pts[j, 0]
                        dy = pts[i, 1] - pts[j, 1]
                        d = math.sqrt(dx * dx + dy * dy)
                        if d < 2.0 - 1e-4:
                            return 1e6 + (2.0 - d) * 1000

                S = compute_S_from_positions(pos_flat, beta)
                return S

            de_bounds = []
            for _ in range(2 * n_pts):
                de_bounds.append((-5.0, 5.0))
            de_bounds.append((0.0, math.pi / 6))

            # Use the best solution as a starting point in the population
            res_de = differential_evolution(
                de_objective, de_bounds,
                seed=42, maxiter=50, popsize=15,
                tol=1e-6, mutation=(0.5, 1.5), recombination=0.9,
                polish=True, init='latinhypercube'
            )

            if res_de.fun < best_joint_S:
                # Verify
                pos_flat = res_de.x[:2 * n_pts]
                beta = res_de.x[2 * n_pts]
                pts = pos_flat.reshape(n_pts, 2)

                min_dist = float('inf')
                for i in range(n_pts):
                    for j in range(i + 1, n_pts):
                        dx = pts[i, 0] - pts[j, 0]
                        dy = pts[i, 1] - pts[j, 1]
                        d = math.sqrt(dx * dx + dy * dy)
                        min_dist = min(min_dist, d)

                if min_dist >= 2.0 - 1e-6:
                    S_exact = compute_S_from_positions(pos_flat, beta)
                    if S_exact < best_joint_S:
                        best_joint_S = S_exact
                        best_joint_x = res_de.x.copy()
        except Exception:
            pass

    # Phase 4: Local polish on the best joint solution
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN and best_joint_S < best_overall_S:
        try:
            # Fine-tune with smaller steps
            x_start = best_joint_x.copy()
            bounds_joint2 = []
            for i in range(2 * n_pts):
                bounds_joint2.append((x_start[i] - 0.5, x_start[i] + 0.5))
            bounds_joint2.append((max(0, x_start[-1] - 0.05), min(math.pi / 6, x_start[-1] + 0.05)))

            res2 = minimize(smooth_objective, x_start, method='L-BFGS-B', bounds=bounds_joint2,
                            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})

            if res2.fun < best_joint_S:
                pos_flat = res2.x[:2 * n_pts]
                beta = res2.x[2 * n_pts]
                pts = pos_flat.reshape(n_pts, 2)

                min_dist = float('inf')
                for i in range(n_pts):
                    for j in range(i + 1, n_pts):
                        dx = pts[i, 0] - pts[j, 0]
                        dy = pts[i, 1] - pts[j, 1]
                        d = math.sqrt(dx * dx + dy * dy)
                        min_dist = min(min_dist, d)

                if min_dist >= 2.0 - 1e-6:
                    S_exact = compute_S_from_positions(pos_flat, beta)
                    if S_exact < best_joint_S:
                        best_joint_S = S_exact
                        best_joint_x = res2.x.copy()
        except Exception:
            pass

    # Use the best result
    if best_joint_S < best_overall_S:
        final_S = best_joint_S
        final_pts = best_joint_x[:2 * n_pts].reshape(n_pts, 2)
        final_beta = best_joint_x[2 * n_pts]
        # Compute center
        normals = np.array([
            [math.cos(final_beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(final_beta + math.pi / 6.0 + k * math.pi / 3.0)]
            for k in range(6)
        ])
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        for nx, ny in normals:
            for px, py in final_pts:
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px * nx + py * ny))
                A_ub.append([nx, ny, -1.0])
                b_ub.append(px * nx + py * ny)
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(None, None), (None, None), (0, None)]
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                final_cx, final_cy = res.x[0], res.x[1]
            else:
                final_cx, final_cy = 0.0, 0.0
        except Exception:
            final_cx, final_cy = 0.0, 0.0
    else:
        final_S = best_overall_S
        final_pts = best_overall_pts
        final_beta = best_overall_beta
        final_cx = best_overall_cx
        final_cy = best_overall_cy

    if final_S == float('inf') or final_S > 10:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Prepare output
    inner = []
    angle_deg = math.degrees(final_beta)
    for x, y in final_pts:
        inner.append([float(x), float(y), angle_deg])

    center = [float(final_cx), float(final_cy)]
    side = float(final_S)
    rotation = angle_deg

    return inner, center, side, rotation
