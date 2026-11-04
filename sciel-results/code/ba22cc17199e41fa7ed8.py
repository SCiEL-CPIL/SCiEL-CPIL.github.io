import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar, fmin_l_bfgs_b

N = 12
DEADLINE = 115.0

_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))
_EDGE_NORMALS = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i+1)%6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i+1)%6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, _ex
    _norm = math.sqrt(_nx*_nx + _ny*_ny)
    _EDGE_NORMALS[_i] = [_nx / _norm, _ny / _norm]

def construct():
    t0 = time.time()
    
    unit_hex_verts = _UNIT_HEX_VERTS
    edge_normals = _EDGE_NORMALS
    sqrt3 = math.sqrt(3)
    
    _ux = unit_hex_verts[:, 0]
    _uy = unit_hex_verts[:, 1]
    _enx = edge_normals[:, 0]
    _eny = edge_normals[:, 1]

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

    def compute_min_outer_side(positions, angles_deg):
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

        best_side = float('inf')
        best_phi = 0.0
        for phi_deg in range(0, 60, 2):
            phi = np.radians(phi_deg)
            normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            abs_proj = np.abs(proj)
            max_projs = np.max(abs_proj, axis=0)
            side_vals = (2.0 / sqrt3) * max_projs
            s_val = np.max(side_vals)
            if s_val < best_side:
                best_side = s_val
                best_phi = phi_deg

        lo = max(0.0, best_phi - 2.0)
        hi = min(60.0, best_phi + 2.0)
        def eval_phi(phi_deg):
            phi = np.radians(phi_deg)
            normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            abs_proj = np.abs(proj)
            max_projs = np.max(abs_proj, axis=0)
            side_vals = (2.0 / sqrt3) * max_projs
            return np.max(side_vals)

        try:
            res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-8})
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

    # Base configuration
    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3/2.0, 1.5])
    lattice_coords = [
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, -1), (-1, 1), (2, -1), (-2, 1),
        (1, 1), (-1, -1), (2, 0),
    ]
    base_positions = np.zeros((N, 2))
    for idx, (i, j) in enumerate(lattice_coords):
        base_positions[idx] = i * a1 + j * a2
    
    radial_dists = np.linalg.norm(base_positions, axis=1)
    sorted_indices = np.argsort(radial_dists)
    ring_assign = np.zeros(N, dtype=int)
    for rank, idx in enumerate(sorted_indices):
        ring_assign[idx] = rank // 2

    def params_to_config(p):
        global_rot = p[0]
        radial_scale = p[1]
        angular_twist = p[2]
        outer_rot = p[33]
        ring_radial_offsets = p[3:9]
        ring_angular_offsets = p[9:15]
        hex_x_pert = p[15:21]
        hex_y_pert = p[21:27]
        hex_angle_pert = p[27:33]
        
        positions = np.zeros((N, 2))
        angles_deg = np.zeros(N)
        
        for i in range(N):
            base_x, base_y = base_positions[i]
            r = math.sqrt(base_x**2 + base_y**2)
            theta = math.atan2(base_y, base_x) if r > 1e-10 else 0.0
            ring = ring_assign[i]
            new_r = r * radial_scale + ring_radial_offsets[ring]
            new_theta = theta + np.radians(angular_twist * r) + np.radians(ring_angular_offsets[ring]) + np.radians(global_rot)
            x = new_r * math.cos(new_theta) + hex_x_pert[i]
            y = new_r * math.sin(new_theta) + hex_y_pert[i]
            positions[i] = [x, y]
            angles_deg[i] = hex_angle_pert[i]
        
        return positions, angles_deg, outer_rot

    def objective(p):
        positions, angles_deg, outer_rot = params_to_config(p)
        angles_rad = np.radians(angles_deg)
        for i in range(N):
            for j in range(i+1, N):
                h1 = (positions[i, 0], positions[i, 1], angles_rad[i])
                h2 = (positions[j, 0], positions[j, 1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    return 1e6
        side, phi, cx, cy = compute_min_outer_side(positions, angles_deg)
        return side

    p0 = np.zeros(34)
    p0[1] = 1.0
    bounds = [(-180, 180), (0.5, 1.5), (-30, 30)]
    for i in range(6):
        bounds.append((-0.5, 0.5))
    for i in range(6):
        bounds.append((-10, 10))
    for i in range(6):
        bounds.append((-0.3, 0.3))
    for i in range(6):
        bounds.append((-0.3, 0.3))
    for i in range(6):
        bounds.append((-15, 15))
    bounds.append((-30, 30))

    best_side = float('inf')
    best_positions = base_positions.copy()
    best_angles = np.zeros(N)
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    base_side, base_phi, base_cx, base_cy = compute_min_outer_side(base_positions, np.zeros(N))
    best_side = base_side
    best_phi = base_phi
    best_cx = base_cx
    best_cy = base_cy

    n_starts = 6
    for start_idx in range(n_starts):
        if time.time() - t0 > DEADLINE - 15:
            break
        p_init = p0.copy()
        if start_idx > 0:
            p_init[0] = random.uniform(-10, 10)
            p_init[1] = random.uniform(0.85, 1.15)
            p_init[2] = random.uniform(-5, 5)
            for i in range(6):
                p_init[3+i] = random.uniform(-0.15, 0.15)
                p_init[9+i] = random.uniform(-3, 3)
            for i in range(6):
                p_init[15+i] = random.uniform(-0.15, 0.15)
                p_init[21+i] = random.uniform(-0.15, 0.15)
                p_init[27+i] = random.uniform(-5, 5)
        try:
            xopt, fopt, ier = fmin_l_bfgs_b(objective, p_init, pgtol=1e-6,
                                              maxfun=3000, maxiter=800, iprint=0, factr=1e4)
            if fopt < best_side and fopt < 1e5:
                positions, angles_deg, outer_rot = params_to_config(xopt)
                if is_valid_configuration(positions, angles_deg):
                    side, phi, cx, cy = compute_min_outer_side(positions, angles_deg)
                    if side < best_side:
                        best_side = side
                        best_positions = positions.copy()
                        best_angles = angles_deg.copy()
                        best_phi = phi
                        best_cx = cx
                        best_cy = cy
        except Exception:
            pass

    if time.time() - t0 < DEADLINE - 5:
        p_cur = p0.copy()
        cur_radial_dists = np.linalg.norm(best_positions, axis=1)
        p_cur[1] = np.mean(cur_radial_dists) / np.mean(radial_dists) if np.mean(radial_dists) > 1e-10 else 1.0
        T = 0.1
        T_min = 0.0001
        cooling = 0.999
        step = 0.05
        cur_side = best_side
        polish_budget = DEADLINE - (time.time() - t0) - 2
        if polish_budget > 1:
            for _ in range(int(polish_budget * 500)):
                if time.time() - t0 > DEADLINE - 2:
                    break
                idx = random.randint(0, 33)
                p_new = p_cur.copy()
                p_new[idx] += random.gauss(0, step)
                lo_b, hi_b = bounds[idx]
                p_new[idx] = max(lo_b, min(hi_b, p_new[idx]))
                f_new = objective(p_new)
                if f_new < cur_side:
                    p_cur = p_new
                    cur_side = f_new
                    if cur_side < best_side:
                        positions, angles_deg, outer_rot = params_to_config(p_cur)
                        if is_valid_configuration(positions, angles_deg):
                            side, phi, cx, cy = compute_min_outer_side(positions, angles_deg)
                            if side < best_side:
                                best_side = side
                                best_positions = positions.copy()
                                best_angles = angles_deg.copy()
                                best_phi = phi
                                best_cx = cx
                                best_cy = cy
                else:
                    delta = f_new - cur_side
                    if delta < 0: delta = 0
                    if random.random() < math.exp(-delta / T):
                        p_cur = p_new
                        cur_side = f_new
                T *= cooling
                if T < T_min: T = T_min
                step *= 0.9999
                if step < 0.001: step = 0.001

    if not is_valid_configuration(best_positions, best_angles):
        best_positions = base_positions.copy()
        best_angles = np.zeros(N)
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    return inner, center, side, angle
