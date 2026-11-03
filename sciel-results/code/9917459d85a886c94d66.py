import math
import time
import random
import numpy as np
from scipy.optimize import minimize_scalar, minimize

N = 12
DEADLINE = 135.0
SQRT3 = math.sqrt(3)
INV_SQRT3 = 1.0 / SQRT3

# Precompute unit hexagon vertices and edge normals (angle 0)
_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))
_EDGE_NORMALS = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i+1)%6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i+1)%6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, _ex
    _norm = math.sqrt(_nx*_nx + _ny*_ny)
    _EDGE_NORMALS[_i] = [_nx / _norm, _ny / _norm]

_UX = _UNIT_HEX_VERTS[:, 0]
_UY = _UNIT_HEX_VERTS[:, 1]
_ENX = _EDGE_NORMALS[:, 0]
_ENY = _EDGE_NORMALS[:, 1]

def construct():
    t0 = time.time()

    def get_all_vertices(positions, angles_rad):
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        rx = _UX[None, :] * c[:, None] - _UY[None, :] * s[:, None]
        ry = _UX[None, :] * s[:, None] + _UY[None, :] * c[:, None]
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
        v1x = h1[0] + _UX * c1 - _UY * s1
        v1y = h1[1] + _UX * s1 + _UY * c1
        v2x = h2[0] + _UX * c2 - _UY * s2
        v2y = h2[1] + _UX * s2 + _UY * c2
        en1x = _ENX * c1 - _ENY * s1
        en1y = _ENX * s1 + _ENY * c1
        en2x = _ENX * c2 - _ENY * s2
        en2y = _ENX * s2 + _ENY * c2
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

    def compute_min_outer_side(positions, angles_deg, phi_hint=None):
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
            side_vals = (2.0 / SQRT3) * max_projs
            return np.max(side_vals)

        # Coarse scan
        best_side = float('inf')
        best_phi = 0.0
        for phi_deg in range(0, 60, 2):
            s = eval_phi(phi_deg)
            if s < best_side:
                best_side = s
                best_phi = phi_deg

        # Fine tune
        try:
            lo = max(0.0, best_phi - 2.0)
            hi = min(60.0, best_phi + 2.0)
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

    # Generate honeycomb initial configuration
    def generate_honeycomb():
        a1 = np.array([SQRT3, 0.0])
        a2 = np.array([SQRT3/2.0, 1.5])
        lattice_coords = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, -1), (-1, 1), (2, -1), (-2, 1),
            (1, 1), (-1, -1), (2, 0),
        ]
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for idx, (i, j) in enumerate(lattice_coords):
            pos[idx] = i * a1 + j * a2
        return pos, ang

    # Generate compact cluster with optimized spacing
    def generate_compact_cluster():
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        # Start with 1 in center, 6 in first ring, 5 in second ring
        pos[0] = [0, 0]
        for k in range(1, 7):
            angle = 2 * np.pi * k / 6
            r = SQRT3
            pos[k] = [r * math.cos(angle), r * math.sin(angle)]
        for k in range(7, 12):
            angle = 2 * np.pi * (k - 6) / 5 + np.pi/5
            r = 2 * SQRT3
            pos[k] = [r * math.cos(angle), r * math.sin(angle)]
        return pos, ang

    # Generate ring pattern
    def generate_ring_pattern():
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        pos[0] = [0, 0]
        for k in range(1, N):
            angle = 2 * np.pi * k / (N-1)
            r = 2.0
            pos[k] = [r * math.cos(angle), r * math.sin(angle)]
        return pos, ang

    # Generate random cluster with overlap resolution
    def generate_random_cluster():
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        pos[0] = [0, 0]
        for k in range(1, N):
            angle = random.uniform(0, 2*np.pi)
            r = random.uniform(1.8, 3.0)
            pos[k] = [r * math.cos(angle), r * math.sin(angle)]
        for _ in range(100):
            for i in range(N):
                for j in range(i+1, N):
                    dx = pos[i,0] - pos[j,0]
                    dy = pos[i,1] - pos[j,1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < 2.0 and dist > 1e-6:
                        push = (2.0 - dist) / 2.0
                        nx, ny = dx/dist, dy/dist
                        pos[i,0] += push * nx
                        pos[i,1] += push * ny
                        pos[j,0] -= push * nx
                        pos[j,1] -= push * ny
        return pos, ang

    # Generate configuration with varied angles
    def generate_varied_angles():
        pos, ang = generate_honeycomb()
        # Add small random angle perturbations
        for i in range(N):
            ang[i] = random.uniform(-10, 10)
        return pos, ang

    initial_configs = [generate_honeycomb(), generate_ring_pattern(),
                       generate_random_cluster(), generate_random_cluster(),
                       generate_varied_angles(), generate_compact_cluster()]

    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    # Evaluate initial configurations
    for pos_init, ang_init in initial_configs:
        if not is_valid_configuration(pos_init, ang_init):
            continue
        side, phi, cx, cy = compute_min_outer_side(pos_init, ang_init)
        if side < best_side:
            best_side = side
            best_positions = pos_init.copy()
            best_angles = ang_init.copy()
            best_phi = phi
            best_cx = cx
            best_cy = cy

    if best_positions is None:
        best_positions, best_angles = initial_configs[0]
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    # SA with adaptive parameters
    current_positions = best_positions.copy()
    current_angles = best_angles.copy()
    current_side, current_phi, current_cx, current_cy = compute_min_outer_side(current_positions, current_angles)

    T = 0.3
    T_min = 0.00005
    cooling = 0.99995
    step_size = 0.2
    step_decay = 0.99999

    step_count = 0
    max_steps = 300000
    accept_count = 0
    total_count = 0

    sa_time_budget = DEADLINE * 0.55

    while time.time() - t0 < sa_time_budget and step_count < max_steps:
        step_count += 1
        total_count += 1

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

        if not valid:
            T *= cooling
            if T < T_min: T = T_min
            step_size *= step_decay
            if step_size < 0.003: step_size = 0.003
            continue

        new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_positions, new_angles)

        if new_side < current_side:
            current_positions = new_positions
            current_angles = new_angles
            current_side = new_side
            current_phi = new_phi
            current_cx = new_cx
            current_cy = new_cy
            accept_count += 1
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
                accept_count += 1
                if new_side < best_side:
                    best_side = new_side
                    best_positions = new_positions.copy()
                    best_angles = new_angles.copy()
                    best_phi = new_phi
                    best_cx = new_cx
                    best_cy = new_cy

        # Adaptive step size
        if total_count % 400 == 0:
            acc_rate = accept_count / total_count
            if acc_rate < 0.08:
                step_size *= 0.75
                if step_size < 0.003: step_size = 0.003
            elif acc_rate > 0.4:
                step_size *= 1.15
                if step_size > 0.8: step_size = 0.8
            accept_count = 0
            total_count = 0

        T *= cooling
        if T < T_min: T = T_min
        step_size *= step_decay
        if step_size < 0.003: step_size = 0.003

    # Phase 2: Tight Coordinate Descent Polish
    remaining_time = DEADLINE - (time.time() - t0)

    if remaining_time > 4.0:
        def polish_tight_cd(pos, ang, time_limit):
            current_side, _, _, _ = compute_min_outer_side(pos, ang)
            best_side_cd = current_side
            best_pos_cd = pos.copy()
            best_ang_cd = ang.copy()

            t_start = time.time()
            step = 0.02
            min_step = 0.0005

            while time.time() - t_start < time_limit and step > min_step:
                improved = False
                for idx in range(N):
                    for dim in range(3):
                        for direction in [1, -1]:
                            new_pos = best_pos_cd.copy()
                            new_ang = best_ang_cd.copy()

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

                            new_side, _, _, _ = compute_min_outer_side(new_pos, new_ang)

                            if new_side < best_side_cd - 1e-12:
                                best_pos_cd = new_pos
                                best_ang_cd = new_ang
                                best_side_cd = new_side
                                improved = True
                                break

                        if time.time() - t_start > time_limit:
                            return best_side_cd, best_pos_cd, best_ang_cd
                    else:
                        continue
                    break
                else:
                    step *= 0.7
                    continue
                break

            return best_side_cd, best_pos_cd, best_ang_cd

        cd_time = min(remaining_time * 0.8, 12.0)
        cd_side, cd_pos, cd_ang = polish_tight_cd(best_positions, best_angles, cd_time)

        if cd_side < best_side:
            best_side = cd_side
            best_positions = cd_pos
            best_angles = cd_ang
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    # Final verification
    if not is_valid_configuration(best_positions, best_angles):
        best_positions, best_angles = generate_honeycomb()
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])

    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)

    return inner, center, side, angle
