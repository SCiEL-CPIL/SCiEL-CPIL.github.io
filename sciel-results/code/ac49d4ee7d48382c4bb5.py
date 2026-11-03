import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 12
DEADLINE = 145.0  # Safety margin before 150s wall-clock kill

# Precompute unit hexagon vertices and edge normals (angle 0)
_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))
_EDGE_NORMALS = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i + 1) % 6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i + 1) % 6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, _ex
    _norm = math.sqrt(_nx * _nx + _ny * _ny)
    _EDGE_NORMALS[_i] = [_nx / _norm, _ny / _norm]

# Pair index list for collision checks
_PAIRS = [(i, j) for i in range(N) for j in range(i + 1, N)]


def construct():
    t0 = time.time()
    sqrt3 = math.sqrt(3)
    _ux = _UNIT_HEX_VERTS[:, 0]
    _uy = _UNIT_HEX_VERTS[:, 1]
    _enx = _EDGE_NORMALS[:, 0]
    _eny = _EDGE_NORMALS[:, 1]

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
        dist_sq = dx * dx + dy * dy
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
            normal_angles = phi + np.pi / 6 + np.arange(6) * np.pi / 3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            abs_proj = np.abs(proj)
            max_projs = np.max(abs_proj, axis=0)
            side_vals = (2.0 / sqrt3) * max_projs
            return np.max(side_vals)

        # Vectorized coarse scan: evaluate all 60 angles at once
        phi_vals = np.radians(np.arange(0, 60, 1))
        normal_angles = phi_vals[:, None] + np.pi / 6 + np.arange(6) * np.pi / 3
        nx = np.cos(normal_angles)  # (60, 6)
        ny = np.sin(normal_angles)  # (60, 6)
        proj = offsets[None, :, 0:1] * nx[:, None, :] + offsets[None, :, 1:2] * ny[:, None, :]
        abs_proj = np.abs(proj)  # (60, 72, 6)
        max_projs = np.max(abs_proj, axis=1)  # (60, 6)
        side_vals = (2.0 / sqrt3) * max_projs  # (60, 6)
        best_idx = np.argmin(np.max(side_vals, axis=1))
        best_side = float(np.max(side_vals[best_idx]))
        best_phi = float(best_idx)

        # Fine tune with minimize_scalar
        try:
            lo = max(0.0, best_phi - 1.0)
            hi = min(60.0, best_phi + 1.0)
            res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-8})
            if res.success and res.fun < best_side:
                best_side = float(res.fun)
                best_phi = float(res.x)
        except Exception:
            pass

        return best_side, best_phi, center[0], center[1]

    def is_valid_configuration(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        for i, j in _PAIRS:
            h1 = (positions[i][0], positions[i][1], angles_rad[i])
            h2 = (positions[j][0], positions[j][1], angles_rad[j])
            if not hexagons_disjoint_fast(h1, h2):
                return False
        return True

    def make_honeycomb():
        a1 = np.array([sqrt3, 0.0])
        a2 = np.array([sqrt3 / 2.0, 1.5])
        lattice_coords = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, -1), (-1, 1), (2, -1), (-2, 1),
            (1, 1), (-1, -1), (2, 0),
        ]
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for idx, (i, j) in enumerate(lattice_coords):
            positions[idx] = i * a1 + j * a2
        return positions, angles

    def make_ring_config():
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for k in range(6):
            a = k * np.pi / 3
            positions[k, 0] = 2 * sqrt3 * math.cos(a)
            positions[k, 1] = 2 * sqrt3 * math.sin(a)
        positions[6] = [0, 0]
        for k in range(5):
            a = k * 2 * np.pi / 5 + np.pi / 5
            positions[7 + k, 0] = 3.5 * sqrt3 * math.cos(a)
            positions[7 + k, 1] = 3.5 * sqrt3 * math.sin(a)
        return positions, angles

    def make_two_rows():
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for k in range(6):
            positions[k, 0] = k * 2 * sqrt3
            positions[k, 1] = 0
            positions[k + 6, 0] = k * 2 * sqrt3 + sqrt3 / 2
            positions[k + 6, 1] = 3.0
        return positions, angles

    def smooth_global_objective(params, beta=40.0):
        """Smooth objective: log-sum-exp outer side + smooth penetration penalty.

        Minkowski-sum inspired: the separating-axis gaps between hexagon pairs
        (computed on the dual hexagonal lattice normals) give a differentiable
        collision measure; the outer side uses a smooth max over vertex
        projections. Both are differentiable so L-BFGS-B can descend directly.
        """
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(N):
            pos[i, 0] = params[3 * i]
            pos[i, 1] = params[3 * i + 1]
            ang[i] = params[3 * i + 2]
        angles_rad = np.radians(ang)

        # ---- Smooth outer side (log-sum-exp over max vertex projections) ----
        all_verts = get_all_vertices(pos, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center

        def phi_side(phi_deg):
            phi = np.radians(phi_deg)
            normal_angles = phi + np.pi / 6 + np.arange(6) * np.pi / 3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            abs_proj = np.abs(proj)
            max_val = np.max(abs_proj, axis=0)
            shifted = abs_proj - max_val[None, :]
            lse = max_val + (1.0 / beta) * np.log(np.sum(np.exp(beta * shifted), axis=0) / len(offsets))
            side_vals = (2.0 / sqrt3) * lse
            return float(np.max(side_vals))

        best_side = float('inf')
        best_phi = 0.0
        for phi_deg in range(0, 60, 3):
            s = phi_side(phi_deg)
            if s < best_side:
                best_side = s
                best_phi = phi_deg
        # refine phi
        try:
            lo = max(0.0, best_phi - 3.0)
            hi = min(60.0, best_phi + 3.0)
            res = minimize_scalar(phi_side, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-6, 'maxiter': 40})
            if res.success and res.fun < best_side:
                best_side = float(res.fun)
                best_phi = float(res.x)
        except Exception:
            pass

        # ---- Smooth collision penalty (separating-axis gap, log-sum-exp) ----
        c_all = np.cos(angles_rad)
        s_all = np.sin(angles_rad)
        # rotated edge normals per hexagon: (N, 6, 2)
        enx_r = _enx[None, :] * c_all[:, None] - _eny[None, :] * s_all[:, None]
        eny_r = _enx[None, :] * s_all[:, None] + _eny[None, :] * c_all[:, None]
        # vertex offsets (N, 6, 2)
        vx_r = _ux[None, :] * c_all[:, None] - _uy[None, :] * s_all[:, None]
        vy_r = _ux[None, :] * s_all[:, None] + _uy[None, :] * c_all[:, None]

        penalty = 0.0
        for i, j in _PAIRS:
            # candidate normals: 12 (6 from each)
            nx = np.concatenate([enx_r[i], enx_r[j]])
            ny = np.concatenate([eny_r[i], eny_r[j]])
            # vertex world coords
            p1x = pos[i, 0] + vx_r[i]
            p1y = pos[i, 1] + vy_r[i]
            p2x = pos[j, 0] + vx_r[j]
            p2y = pos[j, 1] + vy_r[j]
            # projection onto each normal (12,)
            proj1 = p1x * nx + p1y * ny
            proj2 = p2x * nx + p2y * ny
            # gap = max1 - min2 ; penetration = max(0, -gap)
            gap = proj1.max() - proj2.min()
            # smooth max of (0, -gap) using log-sum-exp
            a = np.logaddexp(0.0, -gap)  # log(exp(0)+exp(-gap)) = smooth max(0,-gap)
            penalty += float(a)

        return best_side + 5.0 * penalty

    # ---------------- Initial configurations ----------------
    inits = [make_honeycomb(), make_ring_config(), make_two_rows()]
    best_side = float('inf')
    best_positions = inits[0][0].copy()
    best_angles = inits[0][1].copy()
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    for pos_init, ang_init in inits:
        s, phi, cx, cy = compute_min_outer_side(pos_init, ang_init)
        if s < best_side:
            best_side = s
            best_positions = pos_init.copy()
            best_angles = ang_init.copy()
            best_phi = phi
            best_cx = cx
            best_cy = cy

    # ---------------- Phase 1: Simulated Annealing ----------------
    sa_time_budget = DEADLINE * 0.55
    current_positions = best_positions.copy()
    current_angles = best_angles.copy()
    current_side, current_phi, current_cx, current_cy = compute_min_outer_side(current_positions, current_angles)
    if current_side < best_side:
        best_side = current_side
        best_positions = current_positions.copy()
        best_angles = current_angles.copy()
        best_phi = current_phi
        best_cx = current_cx
        best_cy = current_cy

    T = 0.5
    T_min = 0.0001
    cooling = 0.9999
    step_size = 0.3
    step_decay = 0.99998
    step_count = 0
    max_steps = 200000
    accept_count = 0
    total_count = 0

    while time.time() - t0 < sa_time_budget and step_count < max_steps:
        step_count += 1
        total_count += 1
        idx = random.randint(0, N - 1)
        new_positions = current_positions.copy()
        new_angles = current_angles.copy()
        new_positions[idx, 0] += random.gauss(0, step_size)
        new_positions[idx, 1] += random.gauss(0, step_size)
        new_angles[idx] += random.gauss(0, 5.0)

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
            T = max(T_min, T * cooling)
            step_size = max(0.005, step_size * step_decay)
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
            delta = max(0.0, new_side - current_side)
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

        if total_count % 500 == 0:
            acc_rate = accept_count / total_count
            if acc_rate < 0.1:
                step_size = max(0.005, step_size * 0.8)
            elif acc_rate > 0.5:
                step_size = min(1.0, step_size * 1.2)
            accept_count = 0
            total_count = 0

        T = max(T_min, T * cooling)
        step_size = max(0.005, step_size * step_decay)

    # ---------------- Phase 2: L-BFGS-B on smooth global objective ----------------
    def polish_smooth_lbfgsb(pos, ang, time_limit):
        t_start = time.time()
        init_params = np.zeros(3 * N)
        for i in range(N):
            init_params[3 * i] = pos[i, 0]
            init_params[3 * i + 1] = pos[i, 1]
            init_params[3 * i + 2] = ang[i]

        # multiple random perturbations to escape local minima
        candidates = [init_params.copy()]
        for k in range(3):
            p = init_params.copy()
            p[:2 * N] += random.gauss(0, 0.15, size=2 * N)
            p[2 * N::3] += random.gauss(0, 3.0, size=N)
            candidates.append(p)

        best_res = None
        for cand in candidates:
            if time.time() - t_start > time_limit - 1.0:
                break
            try:
                res = minimize(smooth_global_objective, cand, method='L-BFGS-B',
                               options={'maxiter': 60, 'gtol': 1e-5, 'ftol': 1e-7,
                                        'maxls': 30})
                if best_res is None or res.fun < best_res.fun:
                    best_res = res
            except Exception:
                continue

        if best_res is not None:
            new_pos = np.zeros((N, 2))
            new_ang = np.zeros(N)
            for i in range(N):
                new_pos[i, 0] = best_res.x[3 * i]
                new_pos[i, 1] = best_res.x[3 * i + 1]
                new_ang[i] = best_res.x[3 * i + 2]
            if is_valid_configuration(new_pos, new_ang):
                new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_pos, new_ang)
                return new_side, new_pos, new_ang, new_phi, new_cx, new_cy
        return None

    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 8.0:
        lbfgsb_time = min(remaining_time * 0.35, 30.0)
        res = polish_smooth_lbfgsb(best_positions, best_angles, lbfgsb_time)
        if res is not None:
            s, pos, ang, phi, cx, cy = res
            if s < best_side:
                best_side = s
                best_positions = pos
                best_angles = ang
                best_phi = phi
                best_cx = cx
                best_cy = cy

    # ---------------- Phase 3: Coordinate Descent Polish ----------------
    def polish_coordinate_descent(pos, ang, time_limit, step0, min_step, shrink):
        current_side, _, _, _ = compute_min_outer_side(pos, ang)
        best_side_cd = current_side
        best_pos_cd = pos.copy()
        best_ang_cd = ang.copy()
        t_start = time.time()
        step = step0
        while time.time() - t_start < time_limit and step > min_step:
            improved = False
            for idx in range(N):
                for dim in range(3):
                    for direction in (1, -1):
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
                            if j == idx:
                                continue
                            h_other = (new_pos[j, 0], new_pos[j, 1], angles_rad_new[j])
                            if not hexagons_disjoint_fast(h_moved, h_other):
                                valid = False
                                break
                        if not valid:
                            continue
                        new_side, _, _, _ = compute_min_outer_side(new_pos, new_ang)
                        if new_side < best_side_cd - 1e-10:
                            best_pos_cd = new_pos
                            best_ang_cd = new_ang
                            best_side_cd = new_side
                            improved = True
                            break
                    if time.time() - t_start > time_limit:
                        return best_side_cd, best_pos_cd, best_ang_cd
                    if improved:
                        break
                if time.time() - t_start > time_limit:
                    return best_side_cd, best_pos_cd, best_ang_cd
                if improved:
                    break
            if not improved:
                step *= shrink
        return best_side_cd, best_pos_cd, best_ang_cd

    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 4.0:
        cd_time = min(remaining_time * 0.5, 20.0)
        cd_side, cd_pos, cd_ang = polish_coordinate_descent(best_positions, best_angles, cd_time, 0.02, 0.0005, 0.7)
        if cd_side < best_side:
            best_side = cd_side
            best_positions = cd_pos
            best_angles = cd_ang
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 2.0:
        tight_time = min(remaining_time * 0.6, 15.0)
        t_side, t_pos, t_ang = polish_coordinate_descent(best_positions, best_angles, tight_time, 0.004, 0.0001, 0.6)
        if t_side < best_side:
            best_side = t_side
            best_positions = t_pos
            best_angles = t_ang
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    # ---------------- Final verification ----------------
    if not is_valid_configuration(best_positions, best_angles):
        best_positions, best_angles = make_honeycomb()
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])

    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)

    return inner, center, side, angle
