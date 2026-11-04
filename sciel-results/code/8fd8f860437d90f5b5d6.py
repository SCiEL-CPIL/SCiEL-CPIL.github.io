import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 12
DEADLINE = 145.0  # Safety margin before 150s wall-clock kill
SQRT3 = math.sqrt(3)

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
    sqrt3 = SQRT3
    
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

    def compute_min_outer_side_fast(positions, angles_deg):
        """Fast evaluation: only check a few phi values, no scalar optimization."""
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center

        # Check 12 evenly spaced phi values
        phi_vals = np.radians(np.arange(0, 60, 5))
        normal_angles = phi_vals[:, None] + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, None, 0:1] * nx[None, :, :] + offsets[:, None, 1:2] * ny[None, :, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / sqrt3) * max_projs
        side_per_phi = np.max(side_vals, axis=1)

        best_idx = np.argmin(side_per_phi)
        best_side = side_per_phi[best_idx]
        best_phi = float(best_idx * 5)

        return best_side, best_phi, center[0], center[1]

    def compute_min_outer_side(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center

        phi_vals = np.radians(np.arange(0, 60, 1))
        normal_angles = phi_vals[:, None] + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, None, 0:1] * nx[None, :, :] + offsets[:, None, 1:2] * ny[None, :, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / sqrt3) * max_projs
        side_per_phi = np.max(side_vals, axis=1)

        best_idx = np.argmin(side_per_phi)
        best_side = side_per_phi[best_idx]
        best_phi = float(best_idx)

        def eval_phi(phi_deg):
            phi = np.radians(phi_deg)
            na = phi + np.pi/6 + np.arange(6) * np.pi/3
            nxx = np.cos(na)
            nyy = np.sin(na)
            pr = offsets[:, 0:1] * nxx[None, :] + offsets[:, 1:2] * nyy[None, :]
            return (2.0 / sqrt3) * np.max(np.abs(pr), axis=0).max()

        try:
            lo = max(0.0, best_phi - 1.0)
            hi = min(60.0, best_phi + 1.0)
            res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-8})
            if res.success and res.fun < best_side:
                best_side = res.fun
                best_phi = res.x
        except Exception:
            pass

        return best_side, best_phi, center[0], center[1]

    def smooth_outer_side_surrogate(params, beta=50.0):
        """
        Smooth surrogate for the objective function with overlap penalty.
        params: [x0, y0, a0, ..., x11, y11, a11, phi]
        Returns smooth approximation of min enclosing hexagon side length + overlap penalty.
        """
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(N):
            pos[i, 0] = params[3*i]
            pos[i, 1] = params[3*i+1]
            ang[i] = params[3*i+2]
        phi_deg = params[-1]
        
        angles_rad = np.radians(ang)
        
        all_verts = get_all_vertices(pos, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        phi = np.radians(phi_deg)
        na = phi + np.pi/6 + np.arange(6) * np.pi/3
        nxx = np.cos(na)
        nyy = np.sin(na)
        pr = offsets[:, 0:1] * nxx[None, :] + offsets[:, 1:2] * nyy[None, :]
        abs_pr = np.abs(pr)
        
        # Log-sum-exp over vertices and directions to approximate max
        flat_abs_pr = abs_pr.flatten()
        max_flat = np.max(flat_abs_pr)
        if max_flat > 700.0 / beta:
            return 1e6
        lse = (1.0 / beta) * np.log(np.sum(np.exp((flat_abs_pr - max_flat) * beta))) + max_flat
        
        base_val = (2.0 / sqrt3) * lse
        
        # Add overlap penalty using smooth approximation
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist_sq = dx*dx + dy*dy
                # Minimum distance for non-overlapping unit hexagons is 2*sqrt(3)
                min_dist = 2.0 * sqrt3
                if dist_sq < min_dist * min_dist * 1.1:
                    # Smooth penalty: grows as distance decreases below min_dist
                    dist = math.sqrt(dist_sq) if dist_sq > 0 else 0.001
                    if dist < min_dist:
                        overlap_penalty += (min_dist - dist) ** 2 * 100.0
        
        return base_val + overlap_penalty

    def is_valid_configuration(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        for i in range(N):
            for j in range(i+1, N):
                h1 = (positions[i][0], positions[i][1], angles_rad[i])
                h2 = (positions[j][0], positions[j][1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    return False
        return True

    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3/2.0, 1.5])

    def make_honeycomb():
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

    def make_ring_config():
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(6):
            theta = i * np.pi / 3
            pos[i] = [2 * sqrt3 * math.cos(theta), 2 * sqrt3 * math.sin(theta)]
        for i in range(6):
            theta = (i + 0.5) * np.pi / 3
            pos[i + 6] = [4 * sqrt3 * math.cos(theta), 4 * sqrt3 * math.sin(theta)]
        return pos, ang

    def make_two_rows():
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(6):
            pos[i] = [i * 2 * sqrt3, 0]
            pos[i + 6] = [i * 2 * sqrt3 + sqrt3, 3.0]
        return pos, ang

    def make_compact_hex():
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        pos[0] = [0, 0]
        for i in range(6):
            theta = i * np.pi / 3
            pos[i + 1] = [2 * sqrt3 * math.cos(theta), 2 * sqrt3 * math.sin(theta)]
        for i in range(5):
            theta = (i + 0.5) * np.pi / 3
            pos[i + 7] = [4 * sqrt3 * math.cos(theta), 4 * sqrt3 * math.sin(theta)]
        return pos, ang

    init_configs = [make_honeycomb(), make_ring_config(), make_two_rows(), make_compact_hex()]

    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    for pos0, ang0 in init_configs:
        if is_valid_configuration(pos0, ang0):
            s, phi, cx, cy = compute_min_outer_side(pos0, ang0)
            if s < best_side:
                best_side = s
                best_positions = pos0.copy()
                best_angles = ang0.copy()
                best_phi = phi
                best_cx = cx
                best_cy = cy

    if best_positions is None:
        best_positions, best_angles = make_honeycomb()
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    # Phase 1: Simulated Annealing with fast evaluation
    sa_time_budget = DEADLINE * 0.55
    current_positions = best_positions.copy()
    current_angles = best_angles.copy()
    current_side, current_phi, current_cx, current_cy = compute_min_outer_side_fast(current_positions, current_angles)

    T = 0.3
    T_min = 0.0001
    cooling = 0.9999
    step_size = 0.3
    step_decay = 0.99998

    step_count = 0
    max_steps = 1000000
    accept_count = 0
    total_count = 0

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
            if step_size < 0.002: step_size = 0.002
            continue

        new_side, new_phi, new_cx, new_cy = compute_min_outer_side_fast(new_positions, new_angles)

        if new_side < current_side:
            current_positions = new_positions
            current_angles = new_angles
            current_side = new_side
            current_phi = new_phi
            current_cx = new_cx
            current_cy = new_cy
            accept_count += 1
            # Only do full evaluation for new best
            if new_side < best_side:
                full_side, full_phi, full_cx, full_cy = compute_min_outer_side(new_positions, new_angles)
                if full_side < best_side:
                    best_side = full_side
                    best_positions = new_positions.copy()
                    best_angles = new_angles.copy()
                    best_phi = full_phi
                    best_cx = full_cx
                    best_cy = full_cy
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
                    full_side, full_phi, full_cx, full_cy = compute_min_outer_side(new_positions, new_angles)
                    if full_side < best_side:
                        best_side = full_side
                        best_positions = new_positions.copy()
                        best_angles = new_angles.copy()
                        best_phi = full_phi
                        best_cx = full_cx
                        best_cy = full_cy

        if total_count % 200 == 0:
            acc_rate = accept_count / total_count
            if acc_rate < 0.05:
                step_size *= 0.85
                if step_size < 0.002: step_size = 0.002
            elif acc_rate > 0.35:
                step_size *= 1.1
                if step_size > 0.6: step_size = 0.6
            accept_count = 0
            total_count = 0

        T *= cooling
        if T < T_min: T = T_min
        step_size *= step_decay
        if step_size < 0.002: step_size = 0.002

    # Phase 2: Smooth Surrogate Optimization (L-BFGS-B)
    remaining_time = DEADLINE - (time.time() - t0)
    
    if remaining_time > 15.0:
        # Prepare initial params for L-BFGS-B
        init_params = np.zeros(3*N + 1)
        for i in range(N):
            init_params[3*i] = best_positions[i, 0]
            init_params[3*i+1] = best_positions[i, 1]
            init_params[3*i+2] = best_angles[i]
        init_params[-1] = best_phi
        
        try:
            # Try multiple beta values
            for beta_val in [30.0, 50.0, 80.0]:
                if time.time() - t0 > DEADLINE - 10:
                    break
                res = minimize(lambda p: smooth_outer_side_surrogate(p, beta=beta_val), 
                             init_params, method='L-BFGS-B',
                             options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-7})
                
                if res.fun < best_side + 0.5:
                    new_pos = np.zeros((N, 2))
                    new_ang = np.zeros(N)
                    for i in range(N):
                        new_pos[i, 0] = res.x[3*i]
                        new_pos[i, 1] = res.x[3*i+1]
                        new_ang[i] = res.x[3*i+2]
                    new_phi = res.x[-1]
                    
                    if is_valid_configuration(new_pos, new_ang):
                        new_side, _, _, _ = compute_min_outer_side(new_pos, new_ang)
                        if new_side < best_side:
                            best_side = new_side
                            best_positions = new_pos
                            best_angles = new_ang
                            best_phi = new_phi
                            _, _, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
                            init_params = res.x.copy()  # Use improved solution for next beta
        except Exception:
            pass

    # Phase 3: High-Precision Coordinate Descent Polish with binary search
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 5.0:
        def polish_coordinate_descent(pos, ang, time_limit):
            current_side, _, _, _ = compute_min_outer_side(pos, ang)
            best_side_cd = current_side
            best_pos_cd = pos.copy()
            best_ang_cd = ang.copy()
            
            t_start = time.time()
            
            def try_move(idx, dim, delta):
                """Try moving coordinate idx, dim by delta. Returns (valid, new_side) or (False, None)."""
                new_pos = best_pos_cd.copy()
                new_ang = best_ang_cd.copy()
                
                if dim == 0:
                    new_pos[idx, 0] += delta
                elif dim == 1:
                    new_pos[idx, 1] += delta
                else:
                    new_ang[idx] += delta
                
                angles_rad_new = np.radians(new_ang)
                h_moved = (new_pos[idx, 0], new_pos[idx, 1], angles_rad_new[idx])
                for j in range(N):
                    if j == idx: continue
                    h_other = (new_pos[j, 0], new_pos[j, 1], angles_rad_new[j])
                    if not hexagons_disjoint_fast(h_moved, h_other):
                        return False, None
                
                new_side, _, _, _ = compute_min_outer_side(new_pos, new_ang)
                return True, new_side

            # Multiple passes with decreasing step sizes
            for pass_num in range(15):
                if time.time() - t_start > time_limit:
                    break
                
                step = 0.1 / (pass_num + 1) ** 1.5
                if step < 0.0001:
                    break
                
                improved = False
                for idx in range(N):
                    if time.time() - t_start > time_limit:
                        break
                    for dim in range(3):
                        # Try positive direction with binary search for optimal step
                        best_delta = 0.0
                        best_new_side = best_side_cd
                        
                        # First check if any positive move helps
                        valid, ns = try_move(idx, dim, step)
                        if valid and ns < best_side_cd:
                            # Binary search for best positive delta
                            lo, hi = 0.0, step
                            for _ in range(8):
                                mid = (lo + hi) / 2
                                v, s = try_move(idx, dim, mid)
                                if v and s < best_new_side:
                                    best_delta = mid
                                    best_new_side = s
                                    lo = mid
                                else:
                                    hi = mid
                        
                        # Try negative direction
                        valid, ns = try_move(idx, dim, -step)
                        if valid and ns < best_new_side:
                            lo, hi = -step, 0.0
                            for _ in range(8):
                                mid = (lo + hi) / 2
                                v, s = try_move(idx, dim, mid)
                                if v and s < best_new_side:
                                    best_delta = mid
                                    best_new_side = s
                                    hi = mid
                                else:
                                    lo = mid
                        
                        if best_delta != 0.0 and best_new_side < best_side_cd - 1e-12:
                            if dim == 0:
                                best_pos_cd[idx, 0] += best_delta
                            elif dim == 1:
                                best_pos_cd[idx, 1] += best_delta
                            else:
                                best_ang_cd[idx] += best_delta
                            best_side_cd = best_new_side
                            improved = True
            
            return best_side_cd, best_pos_cd, best_ang_cd

        cd_time = min(remaining_time * 0.7, 25.0)
        cd_side, cd_pos, cd_ang = polish_coordinate_descent(best_positions, best_angles, cd_time)
        
        if cd_side < best_side:
            best_side = cd_side
            best_positions = cd_pos
            best_angles = cd_ang
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    # Final verification and return
    if not is_valid_configuration(best_positions, best_angles):
        # Fallback to best valid config found in SA or init
        for pos0, ang0 in init_configs:
            if is_valid_configuration(pos0, ang0):
                best_positions = pos0.copy()
                best_angles = ang0.copy()
                best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
                break
        else:
            best_positions, best_angles = make_honeycomb()
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
    
    try:
        angles_rad_final = np.radians(best_angles)
        all_verts_final = get_all_vertices(best_positions, angles_rad_final)
        verts_flat_final = all_verts_final.reshape(-1, 2)
        center_final = np.mean(verts_flat_final, axis=0)
        offsets_final = verts_flat_final - center_final
        
        def exact_side_for_phi(phi_deg):
            phi = np.radians(phi_deg)
            na = phi + np.pi/6 + np.arange(6) * np.pi/3
            nxx = np.cos(na)
            nyy = np.sin(na)
            pr = offsets_final[:, 0:1] * nxx[None, :] + offsets_final[:, 1:2] * nyy[None, :]
            return (2.0 / sqrt3) * np.max(np.abs(pr), axis=0).max()
        
        # Fine tune phi one last time
        best_phi_fine = best_phi
        best_side_fine = exact_side_for_phi(best_phi)
        
        for start_phi in [best_phi - 1.0, best_phi, best_phi + 1.0]:
            lo = max(0.0, start_phi - 1.0)
            hi = min(60.0, start_phi + 1.0)
            try:
                res = minimize_scalar(exact_side_for_phi, bounds=(lo, hi), method='bounded',
                                      options={'xatol': 1e-10})
                if res.success and res.fun < best_side_fine:
                    best_side_fine = res.fun
                    best_phi_fine = res.x
            except Exception:
                pass
        
        if best_side_fine < best_side - 1e-12:
            best_side = best_side_fine
            best_phi = best_phi_fine
            best_cx = float(center_final[0])
            best_cy = float(center_final[1])
    except Exception:
        pass
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
