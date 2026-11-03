import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar, least_squares

N = 12
DEADLINE = 145.0  # Safety margin before 150s wall-clock kill

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

def construct():
    t0 = time.time()

    unit_hex_verts = _UNIT_HEX_VERTS
    edge_normals = _EDGE_NORMALS
    sqrt3 = math.sqrt(3)

    _ux = unit_hex_verts[:, 0]
    _uy = unit_hex_verts[:, 1]
    _enx = edge_normals[:, 0]
    _eny = edge_normals[:, 1]

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

    def compute_min_outer_side(positions, angles_deg):
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
            side_vals = (2.0 / sqrt3) * max_projs
            return np.max(side_vals)

        # Vectorized coarse scan with finer resolution
        phi_vals = np.radians(np.arange(0, 60, 0.5))
        normal_angles = phi_vals[:, None] + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, 0:1, None] * nx[None, :, :] + offsets[:, 1:2, None] * ny[None, :, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / sqrt3) * max_projs
        max_sides = np.max(side_vals, axis=1)

        best_idx = np.argmin(max_sides)
        best_side = float(max_sides[best_idx])
        best_phi = float(best_idx * 0.5)

        try:
            lo = max(0.0, best_phi - 1.5)
            hi = min(60.0, best_phi + 1.5)
            res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-9})
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

    def make_honeycomb(scale=1.0, offset_angle=0.0):
        a1 = np.array([sqrt3, 0.0]) * scale
        a2 = np.array([sqrt3/2.0, 1.5]) * scale
        lattice_coords = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, -1), (-1, 1), (2, -1), (-2, 1),
            (1, 1), (-1, -1), (2, 0),
        ]
        positions = np.zeros((N, 2))
        angles = np.zeros(N) + offset_angle
        for idx, (i, j) in enumerate(lattice_coords):
            positions[idx] = i * a1 + j * a2
        return positions, angles

    def make_ring_config():
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for i in range(N):
            theta = 2 * np.pi * i / N
            r = 2.5
            positions[i, 0] = r * np.cos(theta)
            positions[i, 1] = r * np.sin(theta)
            angles[i] = theta * 180 / np.pi
        return positions, angles

    def make_two_rows():
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for i in range(6):
            positions[i, 0] = i * sqrt3
            positions[i, 1] = 0
            positions[i+6, 0] = (i + 0.5) * sqrt3
            positions[i+6, 1] = 1.5
        return positions, angles

    def make_staggered_grid():
        """Two staggered rows with tighter packing."""
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        dx = sqrt3 * 0.95
        dy = 1.5
        for i in range(6):
            positions[i, 0] = i * dx
            positions[i, 1] = 0
            positions[i+6, 0] = (i + 0.5) * dx
            positions[i+6, 1] = dy
        return positions, angles

    def make_compact_cluster():
        """Tighter honeycomb variant."""
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        a1 = np.array([sqrt3 * 0.98, 0.0])
        a2 = np.array([sqrt3/2.0 * 0.98, 1.5 * 0.98])
        lattice_coords = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, -1), (-1, 1), (2, -1), (-2, 1),
            (1, 1), (-1, -1), (0, 2),
        ]
        for idx, (i, j) in enumerate(lattice_coords):
            positions[idx] = i * a1 + j * a2
        return positions, angles

    initial_configs = [
        make_honeycomb(1.0, 0.0),
        make_honeycomb(1.0, 15.0),
        make_honeycomb(0.9, 0.0),
        make_ring_config(),
        make_two_rows(),
        make_staggered_grid(),
        make_compact_cluster(),
    ]

    best_side = float('inf')
    best_positions = initial_configs[0][0].copy()
    best_angles = initial_configs[0][1].copy()
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    # SA phase
    sa_time_budget = DEADLINE * 0.50
    num_restarts = 4
    restart_time = sa_time_budget / num_restarts

    for restart in range(num_restarts):
        if time.time() - t0 >= sa_time_budget:
            break

        init_pos, init_ang = initial_configs[restart % len(initial_configs)]
        current_positions = init_pos.copy()
        current_angles = init_ang.copy()
        current_side, current_phi, current_cx, current_cy = compute_min_outer_side(current_positions, current_angles)

        if current_side < best_side:
            best_side = current_side
            best_positions = current_positions.copy()
            best_angles = current_angles.copy()
            best_phi = current_phi
            best_cx = current_cx
            best_cy = current_cy

        T = 0.8
        T_min = 0.0001
        cooling = 0.9999
        step_size = 0.4
        step_decay = 0.99995

        step_count = 0
        accept_count = 0
        total_count = 0
        restart_deadline = t0 + (restart + 1) * restart_time

        while time.time() < restart_deadline and time.time() - t0 < sa_time_budget:
            step_count += 1
            total_count += 1

            # Occasionally try moving 2 particles simultaneously
            if random.random() < 0.20:
                idx1 = random.randint(0, N - 1)
                idx2 = random.randint(0, N - 1)
                while idx2 == idx1:
                    idx2 = random.randint(0, N - 1)
                new_positions = current_positions.copy()
                new_angles = current_angles.copy()

                for idx in [idx1, idx2]:
                    dx = random.gauss(0, step_size * 0.7)
                    dy = random.gauss(0, step_size * 0.7)
                    da = random.gauss(0, 4.0)
                    new_positions[idx, 0] += dx
                    new_positions[idx, 1] += dy
                    new_angles[idx] += da

                angles_rad_new = np.radians(new_angles)
                valid = True
                for idx in [idx1, idx2]:
                    h_moved = (new_positions[idx, 0], new_positions[idx, 1], angles_rad_new[idx])
                    for j in range(N):
                        if j == idx:
                            continue
                        h_other = (new_positions[j, 0], new_positions[j, 1], angles_rad_new[j])
                        if not hexagons_disjoint_fast(h_moved, h_other):
                            valid = False
                            break
                    if not valid:
                        break

                if not valid:
                    T *= cooling
                    if T < T_min: T = T_min
                    step_size *= step_decay
                    if step_size < 0.005: step_size = 0.005
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
            else:
                idx = random.randint(0, N - 1)
                new_positions = current_positions.copy()
                new_angles = current_angles.copy()

                dx = random.gauss(0, step_size)
                dy = random.gauss(0, step_size)
                da = random.gauss(0, 6.0)

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
                    if step_size < 0.005: step_size = 0.005
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

            if total_count % 200 == 0:
                acc_rate = accept_count / max(total_count, 1)
                if acc_rate < 0.08:
                    step_size *= 0.7
                    if step_size < 0.003: step_size = 0.003
                elif acc_rate > 0.6:
                    step_size *= 1.3
                    if step_size > 1.2: step_size = 1.2
                accept_count = 0
                total_count = 0

            T *= cooling
            if T < T_min: T = T_min
            step_size *= step_decay
            if step_size < 0.003: step_size = 0.003

    # Polish Phase: use scipy least_squares as a structural alternative to smooth-surrogate L-BFGS-B
    remaining_time = DEADLINE - (time.time() - t0)

    if remaining_time > 8.0:
        # Phase A: Least-squares refinement
        # The idea: minimize the "penetration" of vertices beyond the optimal outer hexagon.
        # We parameterize by (positions, angles, phi) and use residuals that are zero when
        # all vertices are inside the outer hexagon of side S.
        # Residual for vertex v at outer-hex orientation phi: for each of 6 half-planes,
        # residual = max(0, dot(v - center, normal_i) - S*sqrt(3)/2)
        # But we want to minimize S, so we use a different formulation:
        # For a given (pos, ang, phi), compute S_min. Then residual = S_min - target_S
        # where target_S is a running lower bound.
        # Better: directly minimize the 6 support function values.

        def lsq_residuals(params):
            """
            params: [x0,y0,a0, x1,y1,a1, ..., x11,y11,a11, phi] (37 values)
            Returns residuals for each of the 72 vertices projected onto 6 normals.
            The outer side for normal i is (2/sqrt3) * max_j |proj_ij|.
            We want to minimize the max over all 6 normals.
            Residuals: for each (vertex, normal) pair, proj - side*sqrt3/2
            But side varies per normal... Instead, use a fixed target side.
            
            Alternative: minimize sum of (max_proj_i - target)^2 for each of 6 normals.
            This is equivalent to minimizing the outer side in a least-squares sense.
            """
            pos = np.zeros((N, 2))
            ang = np.zeros(N)
            for i in range(N):
                pos[i, 0] = params[3*i]
                pos[i, 1] = params[3*i+1]
                ang[i] = params[3*i+2]
            phi = params[3*N]

            # Check validity
            angles_rad = np.radians(ang)
            for i in range(N):
                for j in range(i+1, N):
                    h1 = (pos[i, 0], pos[i, 1], angles_rad[i])
                    h2 = (pos[j, 0], pos[j, 1], angles_rad[j])
                    if not hexagons_disjoint_fast(h1, h2):
                        return np.ones(72) * 10.0

            all_verts = get_all_vertices(pos, angles_rad)
            verts_flat = all_verts.reshape(-1, 2)
            center = np.mean(verts_flat, axis=0)
            offsets = verts_flat - center

            phi_rad = np.radians(phi)
            normal_angles = phi_rad + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            abs_proj = np.abs(proj)
            max_projs = np.max(abs_proj, axis=0)

            # Residuals: each of the 6 max projections minus a common target
            # Target = mean of max_projs (to balance)
            target = np.mean(max_projs)
            residuals = max_projs - target  # 6 values, but we want more structure

            # Expand: for each vertex-normal pair, residual = abs_proj - target
            # This gives 72 residuals that push all vertices toward the boundary
            residuals_full = (abs_proj - target[None, :]).flatten()
            return residuals_full

        def lsq_residuals_v2(params):
            """
            Better formulation: minimize the 6 support values directly.
            residual_i = max_proj_i for each of 6 normals.
            least_squares minimizes sum of squares = sum of max_proj_i^2.
            This is a smooth proxy for the max.
            """
            pos = np.zeros((N, 2))
            ang = np.zeros(N)
            for i in range(N):
                pos[i, 0] = params[3*i]
                pos[i, 1] = params[3*i+1]
                ang[i] = params[3*i+2]
            phi = params[3*N]

            angles_rad = np.radians(ang)
            for i in range(N):
                for j in range(i+1, N):
                    h1 = (pos[i, 0], pos[i, 1], angles_rad[i])
                    h2 = (pos[j, 0], pos[j, 1], angles_rad[j])
                    if not hexagons_disjoint_fast(h1, h2):
                        return np.ones(6) * 100.0

            all_verts = get_all_vertices(pos, angles_rad)
            verts_flat = all_verts.reshape(-1, 2)
            center = np.mean(verts_flat, axis=0)
            offsets = verts_flat - center

            phi_rad = np.radians(phi)
            normal_angles = phi_rad + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            abs_proj = np.abs(proj)
            max_projs = np.max(abs_proj, axis=0)

            return max_projs

        init_params = np.zeros(3*N + 1)
        for i in range(N):
            init_params[3*i] = best_positions[i, 0]
            init_params[3*i+1] = best_positions[i, 1]
            init_params[3*i+2] = best_angles[i]
        init_params[3*N] = best_phi

        try:
            res_lsq = least_squares(lsq_residuals_v2, init_params, method='trf',
                                    max_nfev=5000, xtol=1e-12, ftol=1e-12, gtol=1e-12)

            if res_lsq.success or res_lsq.cost < 1e4:
                new_pos = np.zeros((N, 2))
                new_ang = np.zeros(N)
                for i in range(N):
                    new_pos[i, 0] = res_lsq.x[3*i]
                    new_pos[i, 1] = res_lsq.x[3*i+1]
                    new_ang[i] = res_lsq.x[3*i+2]

                if is_valid_configuration(new_pos, new_ang):
                    new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_pos, new_ang)
                    if new_side < best_side:
                        best_side = new_side
                        best_positions = new_pos
                        best_angles = new_ang
                        best_phi = new_phi
                        best_cx = new_cx
                        best_cy = new_cy
        except Exception:
            pass

    # Phase B: Multi-start least_squares with perturbed initial conditions
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 15.0:
        for trial in range(3):
            if time.time() - t0 >= DEADLINE - 10.0:
                break

            # Perturb the best solution
            pert_pos = best_positions.copy() + np.random.normal(0, 0.15, (N, 2))
            pert_ang = best_angles.copy() + np.random.normal(0, 5.0, N)

            # Check validity of perturbation
            if not is_valid_configuration(pert_pos, pert_ang):
                # Try smaller perturbation
                pert_pos = best_positions.copy() + np.random.normal(0, 0.05, (N, 2))
                pert_ang = best_angles.copy() + np.random.normal(0, 2.0, N)
                if not is_valid_configuration(pert_pos, pert_ang):
                    continue

            pert_side, pert_phi, _, _ = compute_min_outer_side(pert_pos, pert_ang)

            init_params = np.zeros(3*N + 1)
            for i in range(N):
                init_params[3*i] = pert_pos[i, 0]
                init_params[3*i+1] = pert_pos[i, 1]
                init_params[3*i+2] = pert_ang[i]
            init_params[3*N] = pert_phi

            try:
                res_lsq = least_squares(lsq_residuals_v2, init_params, method='trf',
                                        max_nfev=3000, xtol=1e-12, ftol=1e-12, gtol=1e-12)

                if res_lsq.success or res_lsq.cost < 1e4:
                    new_pos = np.zeros((N, 2))
                    new_ang = np.zeros(N)
                    for i in range(N):
                        new_pos[i, 0] = res_lsq.x[3*i]
                        new_pos[i, 1] = res_lsq.x[3*i+1]
                        new_ang[i] = res_lsq.x[3*i+2]

                    if is_valid_configuration(new_pos, new_ang):
                        new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_pos, new_ang)
                        if new_side < best_side:
                            best_side = new_side
                            best_positions = new_pos
                            best_angles = new_ang
                            best_phi = new_phi
                            best_cx = new_cx
                            best_cy = new_cy
            except Exception:
                pass

    # Phase C: Tight coordinate descent polish
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 8.0:
        current_pos = best_positions.copy()
        current_ang = best_angles.copy()
        current_side = best_side
        step = 0.01
        min_step = 0.00005
        t_cd_start = time.time()
        cd_deadline = t0 + DEADLINE - 3.0
        no_improve = 0
        max_no_improve = 8

        while time.time() < cd_deadline and step > min_step:
            improved = False
            for idx in range(N):
                if time.time() > cd_deadline:
                    break
                for dim in range(3):
                    for direction in [1, -1]:
                        new_pos = current_pos.copy()
                        new_ang = current_ang.copy()

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

                        new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_pos, new_ang)

                        if new_side < current_side - 1e-13:
                            current_pos = new_pos
                            current_ang = new_ang
                            current_side = new_side
                            improved = True
                            no_improve = 0
                            if new_side < best_side:
                                best_side = new_side
                                best_positions = new_pos.copy()
                                best_angles = new_ang.copy()
                                best_phi = new_phi
                                best_cx = new_cx
                                best_cy = new_cy
                            break

                    if time.time() > cd_deadline:
                        break
                if time.time() > cd_deadline:
                    break
            if not improved:
                no_improve += 1
                if no_improve >= max_no_improve:
                    step *= 0.5
                    no_improve = 0
                else:
                    step *= 0.7

    # Phase D: Final high-precision phi optimization
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 3.0:
        angles_rad = np.radians(best_angles)
        all_verts = get_all_vertices(best_positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center

        def eval_phi_final(phi_deg):
            phi = np.radians(phi_deg)
            normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            abs_proj = np.abs(proj)
            max_projs = np.max(abs_proj, axis=0)
            side_vals = (2.0 / sqrt3) * max_projs
            return np.max(side_vals)

        # Scan more broadly
        phi_vals = np.arange(0, 60, 0.1)
        best_phi_scan = 0.0
        best_side_scan = float('inf')
        for p in phi_vals:
            s = eval_phi_final(p)
            if s < best_side_scan:
                best_side_scan = s
                best_phi_scan = p

        try:
            lo = max(0.0, best_phi_scan - 2.0)
            hi = min(60.0, best_phi_scan + 2.0)
            res = minimize_scalar(eval_phi_final, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-14, 'maxiter': 1000})
            if res.success and res.fun < best_side:
                best_side = res.fun
                best_phi = res.x
                best_cx = center[0]
                best_cy = center[1]
        except Exception:
            pass

    # Final verification
    if not is_valid_configuration(best_positions, best_angles):
        best_positions = initial_configs[0][0].copy()
        best_angles = initial_configs[0][1].copy()
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])

    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)

    return inner, center, side, angle
