# EVOLVE-BLOCK-START
import numpy as np
import time


def hexagon_packing_12():
    """
    Pack 12 unit regular hexagons in a regular hexagon, minimizing the outer side length s.
    The 12 inner hexagons form a fixed rigid cluster (hexagonal lattice). The only free
    variables are the outer hexagon's rotation angle and center. The minimal containing
    side is computed exactly via a vectorized linear-programming bound, then refined.
    """
    t0 = time.time()
    deadline = t0 + 580.0

    sqrt3 = np.sqrt(3.0)
    S3 = 2.0 * sqrt3  # center-to-center distance between adjacent lattice hexes

    # 12 axial coordinates forming a compact cluster (center + ring1 + 5 of ring2)
    axial = np.array([
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-2, 0), (-1, -1),
    ], dtype=float)

    # Cartesian positions of inner hex centers (all oriented at 0 deg)
    q = axial[:, 0]
    r = axial[:, 1]
    inner_xy = np.column_stack([S3 * q, sqrt3 * (r + 0.5 * q)])  # (12, 2)

    # All 72 vertices of the 12 inner hexes (side 1, angle 0)
    base_ang = np.deg2rad(60.0 * np.arange(6))
    unit_v = np.column_stack([np.cos(base_ang), np.sin(base_ang)])  # (6, 2)
    verts = inner_xy[:, None, :] + unit_v[None, :, :]  # (12, 6, 2)
    V = verts.reshape(72, 2)  # (72, 2)
    V0 = V[0]
    Vc = (V - V0)  # (72, 2)

    # Outer hexagon unit vertices (angle 0) and edge vectors
    uv = unit_v.copy()  # (6, 2)
    e = np.roll(uv, -1, axis=0) - uv  # (6, 2)

    # Precompute: for the angle-0 outer hex, B[i] = -2*cross(e_i, V0)/sqrt3
    B0 = -2.0 * (e[:, 0:1] * V0[1:2] - e[:, 1:2] * V0[0:1]) / sqrt3  # (6,)

    # Precompute: for each angle, the (6, 72) matrix W[i, k] = -2*cross(e_i(theta), Vc_k)/sqrt3
    # We'll compute this on the fly per angle (cheap: 6 x 72 x few ops)

    def min_s(theta_rad, cx, cy):
        """Minimum outer side length for given outer angle theta and center (cx,cy)."""
        a = theta_rad
        ca, sa = np.cos(a), np.sin(a)
        # Rotated unit vertices: R @ uv, where R = [[ca,-sa],[sa,ca]]
        ruv = np.column_stack([ca * uv[:, 0] - sa * uv[:, 1],
                               sa * uv[:, 0] + ca * uv[:, 1]])  # (6, 2)
        re = np.roll(ruv, -1, axis=0) - ruv  # (6, 2)

        # W[i, k] = -2*cross(re_i, Vc_k)/sqrt3
        # cross(re_i, v) = re_i[0]*v[1] - re_i[1]*v[0]
        cross = re[:, 0:1] * Vc[1:2, :] - re[:, 1:2] * Vc[0:1, :]  # (6, 72)
        W = -2.0 * cross / sqrt3  # (6, 72)

        # B[i] = -2*cross(re_i, V0)/sqrt3
        B = -2.0 * (re[:, 0] * V0[1] - re[:, 1] * V0[0]) / sqrt3  # (6,)

        # C[i] = 2*(cx*re_i[1] - cy*re_i[0])/sqrt3
        C = 2.0 * (cx * re[:, 1] - cy * re[:, 0]) / sqrt3  # (6,)

        # s_min = max_i max_k (W[i,k] + B[i] + C[i])
        s = np.max(W + B[:, None] + C[:, None])
        return max(s, 0.0)

    # --- Coarse grid search over angle (0..60) and center ---
    # The cluster is roughly symmetric; center should be near the centroid.
    centroid = np.mean(inner_xy, axis=0)

    # Search angle in [0, 60), center in a box around centroid
    angle_coarse = np.linspace(0, 60, 37)  # 37 points, step ~1.67 deg
    center_range = np.linspace(-2.0, 2.0, 21)  # 21 points, step 0.2

    best_s = np.inf
    best_angle = 0.0
    best_cx = centroid[0]
    best_cy = centroid[1]

    for ang_deg in angle_coarse:
        if time.time() > deadline - 60:
            break
        a = np.deg2rad(ang_deg)
        for cx in center_range:
            for cy in center_range:
                s = min_s(a, cx, cy)
                if s < best_s:
                    best_s = s
                    best_angle = ang_deg
                    best_cx = cx
                    best_cy = cy

    # --- Local refinement (coordinate descent) ---
    if time.time() < deadline - 30:
        steps = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
        for step in steps:
            if time.time() > deadline - 30:
                break
            improved = True
            while improved:
                improved = False
                a = np.deg2rad(best_angle)
                candidates = [
                    (best_angle + step, best_cx, best_cy),
                    (best_angle - step, best_cx, best_cy),
                    (best_angle, best_cx + step, best_cy),
                    (best_angle, best_cx - step, best_cy),
                    (best_angle, best_cx, best_cy + step),
                    (best_angle, best_cx, best_cy - step),
                    (best_angle + step, best_cx + step, best_cy),
                    (best_angle - step, best_cx - step, best_cy),
                    (best_angle + step, best_cx - step, best_cy),
                    (best_angle - step, best_cx + step, best_cy),
                ]
                for ang, cx, cy in candidates:
                    if time.time() > deadline - 30:
                        break
                    s = min_s(np.deg2rad(ang), cx, cy)
                    if s < best_s - 1e-12:
                        best_s = s
                        best_angle = ang
                        best_cx = cx
                        best_cy = cy
                        improved = True

    # --- Final tight refinement ---
    if time.time() < deadline - 10:
        for _ in range(200):
            if time.time() > deadline - 10:
                break
            step = 0.0001
            a = np.deg2rad(best_angle)
            candidates = [
                (best_angle + step, best_cx, best_cy),
                (best_angle - step, best_cx, best_cy),
                (best_angle, best_cx + step, best_cy),
                (best_angle, best_cx - step, best_cy),
                (best_angle, best_cx, best_cy + step),
                (best_angle, best_cx, best_cy - step),
            ]
            for ang, cx, cy in candidates:
                s = min_s(np.deg2rad(ang), cx, cy)
                if s < best_s - 1e-14:
                    best_s = s
                    best_angle = ang
                    best_cx = cx
                    best_cy = cy

    # Construct output
    inner_hex_data = np.column_stack([inner_xy, np.zeros(12)])  # (12, 3)
    outer_hex_data = np.array([best_cx, best_cy, best_angle])

    # Validation
    def _hex_verts(cx, cy, d, angle_deg):
        ang = np.deg2rad(angle_deg) + np.deg2rad(60.0 * np.arange(6))
        return np.column_stack([cx + d * np.cos(ang), cy + d * np.sin(ang)])

    # Check s >= sqrt(12)
    if best_s < np.sqrt(12.0):
        best_s = np.sqrt(12.0) + 1.0

    # Check all inner vertices inside outer hex
    all_verts = verts.reshape(72, 2)
    ov = _hex_verts(best_cx, best_cy, best_s, best_angle)
    valid = True
    for v in all_verts:
        for i in range(6):
            p1 = ov[i]
            p2 = ov[(i + 1) % 6]
            edge = p2 - p1
            to_v = v - p1
            cross_val = edge[0] * to_v[1] - edge[1] * to_v[0]
            if cross_val < -1e-6:
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Increase s slightly to fix boundary violations
        for _ in range(20):
            best_s += 0.01
            ov = _hex_verts(best_cx, best_cy, best_s, best_angle)
            valid = True
            for v in all_verts:
                for i in range(6):
                    p1 = ov[i]
                    p2 = ov[(i + 1) % 6]
                    edge = p2 - p1
                    to_v = v - p1
                    cross_val = edge[0] * to_v[1] - edge[1] * to_v[0]
                    if cross_val < -1e-6:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                break

    return inner_hex_data, outer_hex_data, best_s


# EVOLVE-BLOCK-END
