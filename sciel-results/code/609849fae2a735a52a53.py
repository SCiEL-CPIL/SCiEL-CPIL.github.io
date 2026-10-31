"""Optimized hexagon packing: minimize outer hexagon side length for N unit hexagons.

Structural change: adopt the axial-lattice candidate generation + linear-programming
outer-hexagon computation from the high-scoring inspiration, combined with a smooth
log-sum-exp surrogate for continuous local refinement. The outer hexagon is always
analytically optimal for the current cluster (via LP), and the inner hexagons are
refined using a differentiable surrogate objective.

Uses:
- Axial-coordinate lattice seeds (20 diverse configurations).
- Linear programming (HiGHS) for exact minimal enclosing hexagon at fixed angle.
- minimize_scalar for optimal outer angle refinement.
- Smooth log-sum-exp surrogate for gradient-based inner-hexagon optimization.
- Nelder-Mead polish for final precision.
"""
import numpy as np
import math
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

    # Define candidate point sets (axial coordinates)
    candidates = []

    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    candidates.append(axial_1)

    axial_2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_2)

    axial_3 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ]
    candidates.append(axial_3)

    axial_4 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_4)

    axial_5 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_5)

    axial_6 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_6)

    axial_7 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_7)

    axial_8 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)
    ]
    candidates.append(axial_8)

    axial_9 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (-1, 3), (1, -2)
    ]
    candidates.append(axial_9)

    axial_10 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_10)

    axial_11 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3)
    ]
    candidates.append(axial_11)

    axial_12 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_12)

    axial_13 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, 2)
    ]
    candidates.append(axial_13)

    axial_14 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, -1)
    ]
    candidates.append(axial_14)

    axial_15 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1), (0, -1)
    ]
    candidates.append(axial_15)

    axial_16 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 2)
    ]
    candidates.append(axial_16)

    axial_17 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (1, 2), (-1, 2), (0, 3)
    ]
    candidates.append(axial_17)

    axial_18 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_18)

    axial_19 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 3)
    ]
    candidates.append(axial_19)

    axial_20 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, 1)
    ]
    candidates.append(axial_20)

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def calc_S_for_beta(pts, beta):
        """Compute minimal outer hexagon side length for fixed angle beta using LP."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None

        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))

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
                cx = res.x[0]
                cy = res.x[1]
                S = (D + APOTHEM) / APOTHEM
                return S, (cx, cy), D
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    def smooth_calc_S(pts, beta, cx, cy, temperature=0.1):
        """Smooth log-sum-exp approximation of the max projection."""
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))

        projs = []
        for nx, ny in normals:
            for px, py in pts:
                proj = (px - cx) * nx + (py - cy) * ny
                projs.append(proj)
                projs.append(-proj)

        projs = np.array(projs)
        max_val = np.max(projs)
        if max_val > 100:
            return float('inf'), (cx, cy), float('inf')

        log_sum = np.log(np.sum(np.exp((projs - max_val) / temperature))) + max_val / temperature
        D = temperature * log_sum
        S = (D + APOTHEM) / APOTHEM
        return S, (cx, cy), D

    def optimize_shape(pts):
        """Find optimal outer angle and center for a given set of points."""
        best_S = float('inf')
        best_beta = 0.0
        best_center = None

        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None

        # Coarse scan
        for i in range(37):
            beta = i * (math.pi / 6) / 36
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center

        # Fine scan with minimize_scalar
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0],
                                      bounds=(0, math.pi / 6), method='bounded',
                                      options={'maxiter': 200, 'xatol': 1e-10})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass

        return best_S, best_beta, best_center

    def hexagon_vertices(cx, cy, s, angle_deg):
        """Return 6 vertices of a regular hexagon."""
        a = np.radians(angle_deg)
        angles = a + np.pi / 3 * np.arange(6)
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

    def check_validity(inner_hexagons, outer_center, outer_side, outer_angle):
        """Check if configuration is valid: no overlaps, all inside outer hexagon."""
        ox, oy = outer_center

        # Check overlaps
        for i in range(len(inner_hexagons)):
            for j in range(i + 1, len(inner_hexagons)):
                cx1, cy1, a1 = inner_hexagons[i]
                cx2, cy2, a2 = inner_hexagons[j]
                v1 = hexagon_vertices(cx1, cy1, 1.0, a1)
                v2 = hexagon_vertices(cx2, cy2, 1.0, a2)
                # Simple SAT check
                axes = []
                for v in (v1, v2):
                    n = len(v)
                    for i in range(n):
                        edge = v[(i + 1) % n] - v[i]
                        normal = np.array([-edge[1], edge[0]])
                        norm = np.linalg.norm(normal)
                        if norm > 1e-12:
                            axes.append(normal / norm)
                overlap = True
                for axis in axes:
                    proj1 = np.dot(v1, axis)
                    proj2 = np.dot(v2, axis)
                    if proj1.max() < proj2.min() - EPS or proj2.max() < proj1.min() - EPS:
                        overlap = False
                        break
                if overlap:
                    return False

        # Check containment
        outer_verts = hexagon_vertices(ox, oy, outer_side, outer_angle)
        for cx, cy, ang in inner_hexagons:
            verts = hexagon_vertices(cx, cy, 1.0, ang)
            for v in verts:
                px, py = v
                inside = True
                for i in range(6):
                    v1 = outer_verts[i]
                    v2 = outer_verts[(i + 1) % 6]
                    edge = v2 - v1
                    normal = np.array([-edge[1], edge[0]])
                    d1 = np.dot(normal, np.array([px, py]) - v1)
                    d2 = np.dot(normal, outer_verts[(i + 2) % 6] - v1)
                    if d1 * d2 < 0:
                        inside = False
                        break
                if not inside:
                    return False

        return True

    # Phase 1: Evaluate all axial lattice candidates
    best_S = float('inf')
    best_result = None
    best_pts = None
    best_beta = 0.0
    best_center = None

    for axial in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_shape(pts)
        if S < best_S and center is not None:
            best_S = S
            best_pts = pts
            best_beta = beta
            best_center = center
            # Build result
            inner = [[float(pts[i][0]), float(pts[i][1]), 0.0] for i in range(N)]
            outer_angle_deg = math.degrees(beta) % 360
            best_result = (inner, (float(center[0]), float(center[1])), S, outer_angle_deg)

    if best_result is None:
        # Fallback
        axial_fb = [(0, 0)]
        for k in range(6):
            axial_fb.append((1, k - 2) if k < 3 else (-1, k - 5))
        # Just use simple fallback
        inner = [[0.0, 0.0, 0.0]] * N
        best_result = (inner, (0.0, 0.0), 10.0, 0.0)
        return best_result

    # Phase 2: Smooth optimization to refine inner positions
    # Use the best candidate as starting point
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        try:
            # Convert to parameter vector: 3N (cx, cy, angle for each)
            init_params = np.zeros(3 * N)
            for i in range(N):
                init_params[3 * i] = best_pts[i][0]
                init_params[3 * i + 1] = best_pts[i][1]
                init_params[3 * i + 2] = 0.0  # angle in radians

            def smooth_objective(params, temperature=0.05):
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    return float('inf')
                pts = np.array([[params[3 * i], params[3 * i + 1]] for i in range(N)])
                # Use smooth calc with current best beta and center as starting
                S, _, _ = smooth_calc_S(pts, best_beta, np.array(best_center), temperature)
                return S

            # Try a few temperatures
            for temp in [0.2, 0.1, 0.05, 0.02]:
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    break
                try:
                    res = minimize(smooth_objective, init_params.copy(), args=(temp,),
                                   method='L-BFGS-B',
                                   options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
                    if res.fun < best_S:
                        # Verify with exact LP
                        new_pts = np.array([[res.x[3 * i], res.x[3 * i + 1]] for i in range(N)])
                        new_S, new_beta, new_center = optimize_shape(new_pts)
                        if new_S < best_S and new_center is not None:
                            best_S = new_S
                            best_pts = new_pts
                            best_beta = new_beta
                            best_center = new_center
                            inner = [[float(new_pts[i][0]), float(new_pts[i][1]),
                                      float(math.degrees(res.x[3 * i + 2]) % 360)] for i in range(N)]
                            outer_angle_deg = math.degrees(new_beta) % 360
                            best_result = (inner, (float(new_center[0]), float(new_center[1])),
                                           best_S, outer_angle_deg)
                except Exception:
                    pass
        except Exception:
            pass

    # Phase 3: Nelder-Mead polish
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        try:
            init_params = np.zeros(3 * N)
            for i in range(N):
                init_params[3 * i] = best_pts[i][0]
                init_params[3 * i + 1] = best_pts[i][1]
                init_params[3 * i + 2] = 0.0

            def nm_objective(params):
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    return float('inf')
                pts = np.array([[params[3 * i], params[3 * i + 1]] for i in range(N)])
                S, _, _ = calc_S_for_beta(pts, best_beta)
                return S

            res = minimize(nm_objective, init_params.copy(), method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.fun < best_S:
                new_pts = np.array([[res.x[3 * i], res.x[3 * i + 1]] for i in range(N)])
                new_S, new_beta, new_center = optimize_shape(new_pts)
                if new_S < best_S and new_center is not None:
                    best_S = new_S
                    best_pts = new_pts
                    best_beta = new_beta
                    best_center = new_center
                    inner = [[float(new_pts[i][0]), float(new_pts[i][1]), 0.0] for i in range(N)]
                    outer_angle_deg = math.degrees(new_beta) % 360
                    best_result = (inner, (float(new_center[0]), float(new_center[1])),
                                   best_S, outer_angle_deg)
        except Exception:
            pass

    # Final validation
    if best_result is not None:
        inner, (ox, oy), S, oa = best_result
        if not check_validity(inner, (ox, oy), S, oa):
            # Try to fix by slightly increasing S
            for delta in [0.01, 0.05, 0.1, 0.5]:
                if check_validity(inner, (ox, oy), S + delta, oa):
                    best_result = (inner, (ox, oy), S + delta, oa)
                    break
            else:
                # Fallback to a known valid configuration
                axial_fb = [
                    (0, 0),
                    (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
                    (1, 1), (-1, 2), (-2, 1), (2, -1)
                ]
                pts_fb = axial_to_cartesian(axial_fb)
                S_fb, beta_fb, center_fb = optimize_shape(pts_fb)
                if center_fb is not None:
                    inner_fb = [[float(pts_fb[i][0]), float(pts_fb[i][1]), 0.0] for i in range(N)]
                    best_result = (inner_fb, (float(center_fb[0]), float(center_fb[1])),
                                   S_fb, math.degrees(beta_fb) % 360)

    return best_result


if __name__ == '__main__':
    result = construct()
    print(f"Best S: {result[2]:.6f}")
    print(f"Outer: center={result[1]}, side={result[2]:.6f}, angle={result[3]:.4f}")
    for i, (x, y, a) in enumerate(result[0]):
        print(f"  Inner {i}: ({x:.6f}, {y:.6f}), angle={a:.4f}")
