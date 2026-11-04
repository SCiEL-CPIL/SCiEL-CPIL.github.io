import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 12
DEADLINE = 145.0
T_HARD_LIMIT = 148.0

_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))
_EDGE_NORMALS = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i+1)%6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i+1)%6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, _ex
    _norm = math.sqrt(_nx*_nx + _ny*_ny)
    _EDGE_NORMALS[_i] = [_nx / _norm, _ny / _norm]


def _soft_max(x, beta=50.0):
    m = np.max(x)
    return m + np.log(np.sum(np.exp(beta * (x - m)))) / beta


def construct():
    t0 = time.time()
    sqrt3 = math.sqrt(3)
    two_over_sqrt3 = 2.0 / sqrt3

    _ux = _UNIT_HEX_VERTS[:, 0]
    _uy = _UNIT_HEX_VERTS[:, 1]
    _enx = _EDGE_NORMALS[:, 0]
    _eny = _EDGE_NORMALS[:, 1]

    # --- Geometry helpers ------------------------------------------------

    def get_all_vertices(positions, angles_rad):
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        rx = _ux[None, :] * c[:, None] - _uy[None, :] * s[:, None]
        ry = _ux[None, :] * s[:, None] + _uy[None, :] * c[:, None]
        verts_x = rx + positions[:, 0:1]
        verts_y = ry + positions[:, 1:2]
        return np.stack((verts_x, verts_y), axis=-1)

    def hexagons_disjoint_fast(h1, h2):
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        if dist_sq > 4.5:
            return True
        a1 = h1[2]
        a2 = h2[2]
        c1, s1 = math.cos(a1), math.sin(a1)
        c2, s2 = math.cos(a2), math.sin(a2)
        v1x = h1[0] + _ux * c1 - _uy * s1
        v1y = h1[1] + _ux * s1 + _uy * c1
        v2x = h2[0] + _ux * c2 - _uy * s2
        v2y = h2[1] + _ux * s2 + _uy * c2
        en1x = _enx * c1 - _eny * s1
        en1y = _enx * s1 + _eny * c1
        en2x = _enx * c2 - _eny * s2
        en2y = _enx * s2 + _eny * c2
        for i in range(6):
            nx, ny = en1x[i], en1y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            if p1.max() < p2.min() or p2.max() < p1.min():
                return True
        for i in range(6):
            nx, ny = en2x[i], en2y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            if p1.max() < p2.min() or p2.max() < p1.min():
                return True
        return False

    def penetration_depth(h1, h2):
        """Return positive value if hexagons overlap (approximate), 0 if separated."""
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        if dist_sq > 4.5:
            return 0.0
        a1 = h1[2]
        a2 = h2[2]
        c1, s1 = math.cos(a1), math.sin(a1)
        c2, s2 = math.cos(a2), math.sin(a2)
        v1x = h1[0] + _ux * c1 - _uy * s1
        v1y = h1[1] + _ux * s1 + _uy * c1
        v2x = h2[0] + _ux * c2 - _uy * s2
        v2y = h2[1] + _ux * s2 + _uy * c2
        en1x = _enx * c1 - _eny * s1
        en1y = _enx * s1 + _eny * c1
        en2x = _enx * c2 - _eny * s2
        en2y = _enx * s2 + _eny * c2
        max_pen = 0.0
        for i in range(6):
            nx, ny = en1x[i], en1y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            # max overlap on this axis = max(p1) - min(p2) if positive
            overlap = p1.max() - p2.min()
            if overlap > max_pen:
                max_pen = overlap
        for i in range(6):
            nx, ny = en2x[i], en2y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            overlap = p1.max() - p2.min()
            if overlap > max_pen:
                max_pen = overlap
        return max_pen

    def compute_min_outer_side(positions, angles_deg, fine_tune=True):
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center

        def eval_phi(phi_deg):
            phi = np.radians(phi_deg)
            normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            abs_proj = np.abs(proj)
            max_projs = np.max(abs_proj, axis=0)
            side_vals = two_over_sqrt3 * max_projs
            return np.max(side_vals)

        phi_vals = np.radians(np.arange(0, 60, 1))
        normal_angles = phi_vals[:, None] + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, 0:1, None] * nx[None, :, :] + offsets[:, 1:2, None] * ny[None, :, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = two_over_sqrt3 * max_projs
        max_sides = np.max(side_vals, axis=1)

        best_idx = np.argmin(max_sides)
        best_side = float(max_sides[best_idx])
        best_phi = float(best_idx)

        if fine_tune:
            try:
                lo = max(0.0, best_phi - 1.5)
                hi = min(60.0, best_phi + 1.5)
                res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded',
                                      options={'xatol': 1e-10})
                if res.success and res.fun < best_side:
                    best_side = res.fun
                    best_phi = res.x
            except Exception:
                pass

        return best_side, best_phi, center[0], center[1]

    def is_valid_configuration(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        for i in range(N):
            for j in range(i+1, N):
                h1 = (positions[i][0], positions[i][1], angles_rad[i])
                h2 = (positions[j][0], positions[j][1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    return False
        return True

    def total_penetration(positions, angles_deg):
        """Sum of all pairwise penetration depths. 0 if valid."""
        angles_rad = np.radians(angles_deg)
        total = 0.0
        for i in range(N):
            for j in range(i+1, N):
                h1 = (positions[i][0], positions[i][1], angles_rad[i])
                h2 = (positions[j][0], positions[j][1], angles_rad[j])
                d = penetration_depth(h1, h2)
                if d > 0:
                    total += d
        return total

    # --- Initial configuration generators --------------------------------

    def make_grid_2x3(scale=1.0, rotation_deg=0.0, sx=1.0, sy=1.0, shift_x=0.0, shift_y=0.0):
        a1 = np.array([sqrt3, 0.0]) * scale
        a2 = np.array([sqrt3/2.0, 1.5]) * scale
        a1 = a1 * np.array([sx, 1.0])
        a2 = a2 * np.array([sx, sy])
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        idx = 0
        for j in range(2):
            for i in range(3):
                pos = i * a1 + j * a2
                positions[idx] = pos + np.array([shift_x, shift_y])
                angles[idx] = rotation_deg
                idx += 1
        return positions, angles

    def make_grid_3x2(scale=1.0, rotation_deg=0.0, sx=1.0, sy=1.0, shift_x=0.0, shift_y=0.0):
        a1 = np.array([sqrt3, 0.0]) * scale
        a2 = np.array([sqrt3/2.0, 1.5]) * scale
        a1 = a1 * np.array([sx, 1.0])
        a2 = a2 * np.array([sx, sy])
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        idx = 0
        for j in range(3):
            for i in range(2):
                pos = i * a1 + j * a2
                positions[idx] = pos + np.array([shift_x, shift_y])
                angles[idx] = rotation_deg
                idx += 1
        return positions, angles

    def make_hex_ring(center_pos, radius, rotation_deg=0.0):
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for i in range(N):
            theta = 2 * np.pi * i / N + np.radians(rotation_deg)
            positions[i, 0] = center_pos[0] + radius * math.cos(theta)
            positions[i, 1] = center_pos[1] + radius * math.sin(theta)
            angles[i] = rotation_deg
        return positions, angles

    def make_compact_hex():
        a1 = np.array([sqrt3, 0.0])
        a2 = np.array([sqrt3/2.0, 1.5])
        coords = [
            (0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, -1), (2, 0), (1, 1), (-2, 1), (-2, 0)
        ]
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for idx, (i, j) in enumerate(coords):
            positions[idx] = i * a1 + j * a2
        return positions, angles

    def make_staggered_rows():
        a1 = np.array([sqrt3, 0.0])
        a2 = np.array([sqrt3/2.0, 1.5])
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for i in range(6):
            positions[i] = i * a1
        for i in range(6):
            positions[i+6] = (i + 0.5) * a1 + a2
        return positions, angles

    def make_lattice(scale, coords_list):
        a1 = np.array([sqrt3, 0.0])
        a2 = np.array([sqrt3/2.0, 1.5])
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for idx, (i, j) in enumerate(coords_list):
            positions[idx] = (i * a1 + j * a2) * scale
        return positions, angles

    # --- Smooth surrogate objective with gradient ------------------------

    def smooth_surrogate_and_grad(positions, angles_deg, beta=100.0):
        """
        Smooth outer side length + penalty for overlap, with analytic gradient.
        Returns (value, gradient_vector) where gradient is w.r.t. flattened [pos, ang].
        """
        angles_rad = np.radians(angles_deg)
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        rx = _ux[None, :] * c[:, None] - _uy[None, :] * s[:, None]
        ry = _ux[None, :] * s[:, None] + _uy[None, :] * c[:, None]
        verts_x = rx + positions[:, 0:1]
        verts_y = ry + positions[:, 1:2]
        all_verts = np.stack((verts_x, verts_y), axis=-1)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center

        # Dense phi sampling
        phi_vals = np.radians(np.arange(0, 60, 0.5))
        normal_angles = phi_vals[:, None] + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)

        proj = offsets[:, None, None, 0] * nx[None, :, :, None] + offsets[:, None, None, 1] * ny[None, :, :, None]
        abs_proj = np.abs(proj)

        # Soft-max over vertices
        max_v = abs_proj.max(axis=0, keepdims=True)
        exp_v = np.exp(beta * (abs_proj - max_v))
        sum_exp_v = np.sum(exp_v, axis=0)
        logsumexp_v = max_v.squeeze(0) + np.log(sum_exp_v) / beta
        w_v = exp_v / sum_exp_v[None, :, :]

        # Soft-max over edge directions
        max_e = logsumexp_v.max(axis=1, keepdims=True)
        exp_e = np.exp(beta * (logsumexp_v - max_e))
        sum_exp_e = np.sum(exp_e, axis=1)
        logsumexp_e = max_e.squeeze(1) + np.log(sum_exp_e) / beta
        w_e = exp_e / sum_exp_e[:, None]

        # Soft-max over phi
        max_p = logsumexp_e.max(axis=0, keepdims=True)
        exp_p = np.exp(beta * (logsumexp_e - max_p))
        sum_exp_p = np.sum(exp_p)
        logsumexp_phi = max_p.squeeze(0) + np.log(sum_exp_p) / beta
        w_p = exp_p / sum_exp_p

        obj = two_over_sqrt3 * logsumexp_phi

        # Gradient of outer side
        w_combined = w_p[:, None] * w_e
        sign_proj = np.sign(proj)
        weighted = w_v * w_combined[None, :, :] * sign_proj
        contrib = weighted @ normal  # (72, 2)

        total_contrib = np.sum(contrib, axis=0)
        hex_contrib = np.zeros((N, 2))
        for h in range(N):
            hex_contrib[h] = np.sum(contrib[h*6:(h+1)*6], axis=0)
        grad_pos = two_over_sqrt3 * (hex_contrib - total_contrib / 72.0)

        grad_ang = np.zeros(N)
        for h in range(N):
            ch = c[h]
            sh = s[h]
            local_verts = _UNIT_HEX_VERTS
            dverts = np.array([
                [-sh * local_verts[:, 0] - ch * local_verts[:, 1]],
                [ch * local_verts[:, 0] - sh * local_verts[:, 1]]
            ])
            sum_dverts = np.sum(dverts, axis=1)
            ga = 0.0
            for l in range(6):
                v_idx = h * 6 + l
                d_off = dverts[:, l] - sum_dverts / 72.0
                ga += contrib[v_idx, 0] * d_off[0] + contrib[v_idx, 1] * d_off[1]
            grad_ang[h] = two_over_sqrt3 * ga

        # Penalty gradient for overlap (use a simple repulsive force)
        penalty_val = 0.0
        pen_grad_pos = np.zeros((N, 2))
        pen_grad_ang = np.zeros(N)
        penalty_lambda = 5.0
        for i in range(N):
            for j in range(i+1, N):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist_sq = dx*dx + dy*dy
                if dist_sq < 4.5:
                    # Approximate penetration
                    a1 = angles_rad[i]
                    a2 = angles_rad[j]
                    c1, s1 = math.cos(a1), math.sin(a1)
                    c2, s2 = math.cos(a2), math.sin(a2)
                    v1x = positions[i, 0] + _ux * c1 - _uy * s1
                    v1y = positions[i, 1] + _ux * s1 + _uy * c1
                    v2x = positions[j, 0] + _ux * c2 - _uy * s2
                    v2y = positions[j, 1] + _ux * s2 + _uy * c2
                    en1x = _enx * c1 - _eny * s1
                    en1y = _enx * s1 + _eny * c1
                    en2x = _enx * c2 - _eny * s2
                    en2y = _enx * s2 + _eny * c2
                    max_pen = 0.0
                    best_nx, best_ny = 0, 0
                    for k in range(6):
                        nx_k, ny_k = en1x[k], en1y[k]
                        p1 = v1x * nx_k + v1y * ny_k
                        p2 = v2x * nx_k + v2y * ny_k
                        overlap = p1.max() - p2.min()
                        if overlap > max_pen:
                            max_pen = overlap
                            best_nx, best_ny = nx_k, ny_k
                    for k in range(6):
                        nx_k, ny_k = en2x[k], en2y[k]
                        p1 = v1x * nx_k + v1y * ny_k
                        p2 = v2x * nx_k + v2y * ny_k
                        overlap = p1.max() - p2.min()
                        if overlap > max_pen:
                            max_pen = overlap
                            best_nx, best_ny = nx_k, ny_k
                    if max_pen > 0:
                        penalty_val += penalty_lambda * max_pen
                        pen_grad_pos[i, 0] += penalty_lambda * best_nx
                        pen_grad_pos[i, 1] += penalty_lambda * best_ny
                        pen_grad_pos[j, 0] -= penalty_lambda * best_nx
                        pen_grad_pos[j, 1] -= penalty_lambda * best_ny
                        # Angle gradient (approximate)
                        pen_grad_ang[i] += penalty_lambda * 0.01
                        pen_grad_ang[j] -= penalty_lambda * 0.01

        total_val = obj + penalty_val
        grad = np.concatenate([grad_pos.flatten() + pen_grad_pos.flatten(), grad_ang + pen_grad_ang])
        return total_val, grad

    def smooth_surrogate_obj(positions, angles_deg, beta=100.0):
        """Objective only (faster)."""
        angles_rad = np.radians(angles_deg)
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        rx = _ux[None, :] * c[:, None] - _uy[None, :] * s[:, None]
        ry = _ux[None, :] * s[:, None] + _uy[None, :] * c[:, None]
        verts_x = rx + positions[:, 0:1]
        verts_y = ry + positions[:, 1:2]
        all_verts = np.stack((verts_x, verts_y), axis=-1)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center

        phi_vals = np.radians(np.arange(0, 60, 0.5))
        normal_angles = phi_vals[:, None] + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, None, None, 0] * nx[None, :, :, None] + offsets[:, None, None, 1] * ny[None, :, :, None]
        abs_proj = np.abs(proj)

        max_v = abs_proj.max(axis=0, keepdims=True)
        logsumexp_v = max_v.squeeze(0) + np.log(np.sum(np.exp(beta * (abs_proj - max_v)), axis=0)) / beta
        max_e = logsumexp_v.max(axis=1, keepdims=True)
        logsumexp_e = max_e.squeeze(1) + np.log(np.sum(np.exp(beta * (logsumexp_v - max_e)), axis=1)) / beta
        max_p = logsumexp_e.max(axis=0, keepdims=True)
        logsumexp_phi = max_p.squeeze(0) + np.log(np.sum(np.exp(beta * (logsumexp_e - max_p)), axis=0)) / beta

        # Add penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist_sq = dx*dx + dy*dy
                if dist_sq < 4.5:
                    a1 = angles_rad[i]
                    a2 = angles_rad[j]
                    c1, s1 = math.cos(a1), math.sin(a1)
                    c2, s2 = math.cos(a2), math.sin(a2)
                    v1x = positions[i, 0] + _ux * c1 - _uy * s1
                    v1y = positions[i, 1] + _ux * s1 + _uy * c1
                    v2x = positions[j, 0] + _ux * c2 - _uy * s2
                    v2y = positions[j, 1] + _ux * s2 + _uy * c2
                    en1x = _enx * c1 - _eny * s1
                    en1y = _enx * s1 + _eny * c1
                    en2x = _enx * c2 - _eny * s2
                    en2y = _enx * s2 + _eny * c2
                    max_pen = 0.0
                    for k in range(6):
                        p1 = v1x * en1x[k] + v1y * en1y[k]
                        p2 = v2x * en1x[k] + v2y * en1y[k]
                        ov = p1.max() - p2.min()
                        if ov > max_pen: max_pen = ov
                    for k in range(6):
                        p1 = v1x * en2x[k] + v1y * en2y[k]
                        p2 = v2x * en2x[k] + v2y * en2y[k]
                        ov = p1.max() - p2.min()
                        if ov > max_pen: max_pen = ov
                    if max_pen > 0:
                        penalty += 5.0 * max_pen

        return two_over_sqrt3 * logsumexp_phi + penalty

    # --- Generate initial configs ----------------------------------------

    initial_configs = []
    for s in [0.92, 0.95, 0.98, 1.0, 1.02, 1.05]:
        initial_configs.append(make_grid_2x3(scale=s))
        initial_configs.append(make_grid_3x2(scale=s))
    for rot in [0, 5, 10, 15, 20, 30, 45]:
        initial_configs.append(make_grid_2x3(scale=1.0, rotation_deg=rot))
        initial_configs.append(make_grid_3x2(scale=1.0, rotation_deg=rot))
    for sx in [0.95, 0.98, 1.0, 1.02, 1.05]:
        for sy in [0.95, 0.98, 1.0, 1.02, 1.05]:
            initial_configs.append(make_grid_2x3(scale=1.0, sx=sx, sy=sy))
    for r in [1.8, 2.0, 2.2, 2.5, 2.8, 3.0, 3.5]:
        initial_configs.append(make_hex_ring([0, 0], r))
    initial_configs.append(make_compact_hex())
    initial_configs.append(make_staggered_rows())

    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3/2.0, 1.5])
    lattice_coords_list = [
        [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
         (2, -1), (-2, 1), (1, 1), (-1, -1), (2, 0)],
        [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1),
         (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (2, -1)],
        [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
         (2, -1), (2, 0), (1, 1), (-2, 1), (-2, 0)],
        [(0,0),(1,0),(2,0),(3,0),(4,0),(5,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)],
        [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1),(1,1),(-1,-1),(2,-1),(-2,1),(2,0)],
    ]
    for scale in [0.85, 0.90, 0.95, 1.0, 1.05]:
        for coords in lattice_coords_list:
            initial_configs.append(make_lattice(scale, coords))

    # --- Evaluate initial configs ----------------------------------------

    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    for pos, ang in initial_configs:
        if time.time() - t0 > DEADLINE * 0.15:
            break
        try:
            if total_penetration(pos, ang) > 0.01:
                continue
            side, phi, cx, cy = compute_min_outer_side(pos, ang, fine_tune=False)
            if side < best_side:
                best_side = side
                best_positions = pos.copy()
                best_angles = ang.copy()
                best_phi = phi
                best_cx = cx
                best_cy = cy
        except Exception:
            continue

    if best_positions is None:
        best_positions, best_angles = make_grid_2x3(1.0)
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    # --- Simulated Annealing phase ---------------------------------------

    sa_deadline = t0 + DEADLINE * 0.35
    current_positions = best_positions.copy()
    current_angles = best_angles.copy()
    current_side, current_phi, current_cx, current_cy = compute_min_outer_side(current_positions, current_angles)

    T = 0.3
    T_min = 0.00003
    cooling = 0.99998
    step_size = 0.15
    step_decay = 0.999995
    min_step = 0.001

    iters = 0
    while time.time() < sa_deadline:
        idx = random.randint(0, N - 1)
        new_positions = current_positions.copy()
        new_angles = current_angles.copy()

        dx = random.gauss(0, step_size)
        dy = random.gauss(0, step_size)
        da = random.gauss(0, 3.0)
        new_positions[idx, 0] += dx
        new_positions[idx, 1] += dy
        new_angles[idx] += da

        angles_rad_new = np.radians(new_angles)
        valid = True
        h_moved = (new_positions[idx, 0], new_positions[idx, 1], angles_rad_new[idx])
        for j in range(N):
            if j == idx:
                continue
            h_other = (new_positions[j, 0], new_positions[j, 1], angles_rad_new[j])
            if not hexagons_disjoint_fast(h_moved, h_other):
                valid = False
                break

        if valid:
            new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_positions, new_angles, fine_tune=False)
            if new_side < current_side:
                current_positions = new_positions
                current_angles = new_angles
                current_side = new_side
                current_phi = new_phi
                current_cx = new_cx
                current_cy = new_cy
                if new_side < best_side:
                    best_side = new_side
                    best_positions = new_positions.copy()
                    best_angles = new_angles.copy()
                    best_phi = new_phi
                    best_cx = new_cx
                    best_cy = new_cy
            else:
                delta = new_side - current_side
                if delta < 0: delta = 0
                if random.random() < math.exp(-delta / T):
                    current_positions = new_positions
                    current_angles = new_angles
                    current_side = new_side
                    current_phi = new_phi
                    current_cx = new_cx
                    current_cy = new_cy
                    if new_side < best_side:
                        best_side = new_side
                        best_positions = new_positions.copy()
                        best_angles = new_angles.copy()
                        best_phi = new_phi
                        best_cx = new_cx
                        best_cy = new_cy

        T *= cooling
        if T < T_min: T = T_min
        step_size *= step_decay
        if step_size < min_step: step_size = min_step
        iters += 1
        if iters % 5000 == 0 and time.time() > T_HARD_LIMIT:
            break

    # --- Gradient-based optimization with penalized smooth surrogate -----

    remaining = DEADLINE - (time.time() - t0)
    if remaining > 25.0:
        grad_deadline = t0 + remaining * 0.50
        x0 = np.concatenate([best_positions.flatten(), best_angles])

        def obj_and_grad(x):
            pos = x[:2*N].reshape(N, 2)
            ang = x[2*N:]
            return smooth_surrogate_and_grad(pos, ang, beta=100.0)

        # Multiple gradient optimizations with different betas
        for beta_val in [80.0, 100.0, 150.0]:
            if time.time() > grad_deadline:
                break
            try:
                def og(x, b=beta_val):
                    pos = x[:2*N].reshape(N, 2)
                    ang = x[2*N:]
                    return smooth_surrogate_and_grad(pos, ang, beta=b)

                res = minimize(og, x0, jac=True, method='L-BFGS-B',
                               options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
                if res.success:
                    new_pos = res.x[:2*N].reshape(N, 2)
                    new_ang = res.x[2*N:]
                    if total_penetration(new_pos, new_ang) < 0.005:
                        new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_pos, new_ang)
                        if new_side < best_side:
                            best_side = new_side
                            best_positions = new_pos.copy()
                            best_angles = new_ang.copy()
                            best_phi = new_phi
                            best_cx = new_cx
                            best_cy = new_cy
                            x0 = res.x.copy()
            except Exception:
                pass

        # Perturbed restarts
        if time.time() < grad_deadline:
            for restart in range(10):
                if time.time() > grad_deadline:
                    break
                x0_pert = x0.copy()
                x0_pert[:2*N] += random.gauss(0, 0.05, (2*N,))
                x0_pert[2*N:] += random.gauss(0, 1.5, (N,))
                try:
                    def og2(x):
                        pos = x[:2*N].reshape(N, 2)
                        ang = x[2*N:]
                        return smooth_surrogate_and_grad(pos, ang, beta=100.0)
                    res = minimize(og2, x0_pert, jac=True, method='L-BFGS-B',
                                   options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 40})
                    if res.success:
                        new_pos = res.x[:2*N].reshape(N, 2)
                        new_ang = res.x[2*N:]
                        if total_penetration(new_pos, new_ang) < 0.005:
                            new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_pos, new_ang)
                            if new_side < best_side:
                                best_side = new_side
                                best_positions = new_pos.copy()
                                best_angles = new_ang.copy()
                                best_phi = new_phi
                                best_cx = new_cx
                                best_cy = new_cy
                                x0 = res.x.copy()
                except Exception:
                    pass

    # --- Coordinate descent polish ---------------------------------------

    remaining = DEADLINE - (time.time() - t0)
    if remaining > 10.0:
        cd_deadline = t0 + remaining * 0.55

        # Multi-stage coordinate descent
        for stage, (init_step, min_step_cd) in enumerate([(0.05, 0.00002), (0.01, 0.00001), (0.003, 0.000005)]):
            step = init_step
            stage_deadline = cd_deadline if stage == 2 else t0 + (cd_deadline - t0) * (0.3 + 0.35 * stage)
            while time.time() < stage_deadline and step > min_step_cd:
                improved = False
                for idx in range(N):
                    if time.time() > stage_deadline:
                        break
                    for dim in range(3):
                        if time.time() > stage_deadline:
                            break
                        for direction in [1, -1]:
                            new_pos = best_positions.copy()
                            new_ang = best_angles.copy()
                            if dim == 0:
                                new_pos[idx, 0] += direction * step
                            elif dim == 1:
                                new_pos[idx, 1] += direction * step
                            else:
                                new_ang[idx] += direction * step

                            angles_rad_new = np.radians(new_ang)
                            valid = True
                            h_moved = (new_pos[idx, 0], new_pos[idx, 1], angles_rad_new[idx])
                            for j in range(N):
                                if j == idx: continue
                                h_other = (new_pos[j, 0], new_pos[j, 1], angles_rad_new[j])
                                if not hexagons_disjoint_fast(h_moved, h_other):
                                    valid = False
                                    break
                            if not valid:
                                continue

                            new_side, _, _, _ = compute_min_outer_side(new_pos, new_ang, fine_tune=False)
                            if new_side < best_side - 1e-12:
                                best_positions = new_pos
                                best_angles = new_ang
                                best_side = new_side
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if not improved:
                    step *= 0.5
                else:
                    step = max(step * 0.8, min_step_cd)

        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    # --- Final high-precision phi optimization ----------------------------

    remaining2 = DEADLINE - (time.time() - t0)
    if remaining2 > 3.0:
        def eval_final_phi(phi_deg):
            phi = np.radians(phi_deg)
            angles_rad = np.radians(best_angles)
            all_verts = get_all_vertices(best_positions, angles_rad)
            verts_flat = all_verts.reshape(-1, 2)
            center = np.mean(verts_flat, axis=0)
            offsets = verts_flat - center
            normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            abs_proj = np.abs(proj)
            max_projs = np.max(abs_proj, axis=0)
            side_vals = two_over_sqrt3 * max_projs
            return np.max(side_vals)

        try:
            best_phi_val = best_side
            best_phi_ang = best_phi
            for start in [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, best_phi]:
                lo = max(0.0, start - 12.0)
                hi = min(60.0, start + 12.0)
                res = minimize_scalar(eval_final_phi, bounds=(lo, hi), method='bounded',
                                      options={'xatol': 1e-12})
                if res.success and res.fun < best_phi_val:
                    best_phi_val = res.fun
                    best_phi_ang = res.x
            if best_phi_val < best_side:
                best_side = best_phi_val
                best_phi = best_phi_ang
                angles_rad = np.radians(best_angles)
                all_verts = get_all_vertices(best_positions, angles_rad)
                verts_flat = all_verts.reshape(-1, 2)
                center = np.mean(verts_flat, axis=0)
                best_cx = center[0]
                best_cy = center[1]
        except Exception:
            pass

    # --- Final validation -------------------------------------------------

    if not is_valid_configuration(best_positions, best_angles):
        # Try to recover
        found = False
        for pos, ang in initial_configs[:20]:
            if is_valid_configuration(pos, ang):
                side, phi, cx, cy = compute_min_outer_side(pos, ang)
                if side < best_side:
                    best_side = side
                    best_positions = pos.copy()
                    best_angles = ang.copy()
                    best_phi = phi
                    best_cx = cx
                    best_cy = cy
                    found = True
                    break
        if not found:
            best_positions, best_angles = make_grid_2x3(1.0)
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    final_side, final_phi, final_cx, final_cy = compute_min_outer_side(best_positions, best_angles)
    if final_side < best_side:
        best_side = final_side
        best_phi = final_phi
        best_cx = final_cx
        best_cy = final_cy

    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])

    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)

    return inner, center, side, angle
