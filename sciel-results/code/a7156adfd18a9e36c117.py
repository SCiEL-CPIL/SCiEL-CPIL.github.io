import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 12
DEADLINE = 144.0  # Safety margin before 150s wall-clock kill

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
    
    # Precompute flat arrays for speed
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
        
        # Vectorized coarse scan: evaluate 120 angles (0.5° step) at once
        phi_vals = np.radians(np.arange(0, 60, 0.5))
        normal_angles = phi_vals[:, None] + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)  # (120, 6)
        ny = np.sin(normal_angles)  # (120, 6)
        proj = offsets[None, :, 0:1] * nx[:, None, :] + offsets[None, :, 1:2] * ny[:, None, :]
        abs_proj = np.abs(proj)  # (120, 72, 6)
        max_projs = np.max(abs_proj, axis=1)  # (120, 6)
        side_vals = (2.0 / sqrt3) * max_projs  # (120, 6)
        best_idx = np.argmin(np.max(side_vals, axis=1))
        best_side = np.max(side_vals[best_idx])
        best_phi = float(best_idx * 0.5)
        
        # Fine tune with minimize_scalar
        try:
            lo = max(0.0, best_phi - 0.5)
            hi = min(60.0, best_phi + 0.5)
            
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

    def make_honeycomb():
        a1 = np.array([sqrt3, 0.0])
        a2 = np.array([sqrt3/2.0, 1.5])
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
            positions[k, 0] = 2*sqrt3 * math.cos(a)
            positions[k, 1] = 2*sqrt3 * math.sin(a)
        positions[6] = [0, 0]
        for k in range(5):
            a = k * 2*np.pi/5 + np.pi/5
            positions[7+k, 0] = 3.5*sqrt3 * math.cos(a)
            positions[7+k, 1] = 3.5*sqrt3 * math.sin(a)
        return positions, angles

    def make_two_rows():
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for k in range(6):
            positions[k, 0] = k * 2*sqrt3
            positions[k, 1] = 0
            positions[k+6, 0] = k * 2*sqrt3 + sqrt3/2
            positions[k+6, 1] = 3.0
        return positions, angles

    def make_triangular():
        # Triangular arrangement: 3-4-3 or similar
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        # Row 1: 3 hexagons
        for k in range(3):
            positions[k, 0] = k * 2*sqrt3
            positions[k, 1] = 0
        # Row 2: 4 hexagons
        for k in range(4):
            positions[3+k, 0] = k * 2*sqrt3 + sqrt3/2
            positions[3+k, 1] = 3.0
        # Row 3: 3 hexagons
        for k in range(3):
            positions[7+k, 0] = k * 2*sqrt3
            positions[7+k, 1] = 6.0
        return positions, angles

    def make_compact_hex():
        # More compact: center + 6 around + 5 more
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        positions[0] = [0, 0]
        for k in range(6):
            a = k * np.pi / 3
            positions[1+k, 0] = 2*sqrt3 * math.cos(a)
            positions[1+k, 1] = 2*sqrt3 * math.sin(a)
        # 5 more at distance ~3*sqrt3
        for k in range(5):
            a = k * 2*np.pi/5
            positions[7+k, 0] = 3*sqrt3 * math.cos(a)
            positions[7+k, 1] = 3*sqrt3 * math.sin(a)
        return positions, angles

    # Evaluate multiple initial configurations
    inits = [make_honeycomb(), make_ring_config(), make_two_rows(), make_triangular(), make_compact_hex()]
    
    best_side = float('inf')
    best_positions = None
    best_angles = None
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

    # Strategy: Run SA for 60% of the time, then Polish for the rest
    sa_time_budget = DEADLINE * 0.60
    
    # Start SA from best initial configuration
    current_positions = best_positions.copy()
    current_angles = best_angles.copy()
    current_side = best_side
    current_phi = best_phi
    current_cx = best_cx
    current_cy = best_cy

    T = 0.3
    T_min = 0.00005
    cooling = 0.99995
    step_size = 0.25
    step_decay = 0.99999
    
    step_count = 0
    max_steps = 300000
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
        da = random.gauss(0, 4.0)
        
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
        if total_count % 300 == 0:
            acc_rate = accept_count / total_count
            if acc_rate < 0.15:
                step_size *= 0.85
                if step_size < 0.003: step_size = 0.003
            elif acc_rate > 0.45:
                step_size *= 1.15
                if step_size > 0.8: step_size = 0.8
            accept_count = 0
            total_count = 0
            
        T *= cooling
        if T < T_min: T = T_min
        step_size *= step_decay
        if step_size < 0.003: step_size = 0.003

    # Polish Phase
    remaining_time = DEADLINE - (time.time() - t0)
    
    if remaining_time > 2.0:
        # 1. Coordinate Descent Polish
        def polish_coordinate_descent(pos, ang, time_limit):
            current_side, _, _, _ = compute_min_outer_side(pos, ang)
            best_side_cd = current_side
            best_pos_cd = pos.copy()
            best_ang_cd = ang.copy()
            
            t_start = time.time()
            step = 0.03
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
                            
                            if new_side < best_side_cd - 1e-10:
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
                    step *= 0.8
                    continue
                break
            
            return best_side_cd, best_pos_cd, best_ang_cd

        cd_time = min(remaining_time * 0.35, 12.0)
        cd_side, cd_pos, cd_ang = polish_coordinate_descent(best_positions, best_angles, cd_time)
        
        if cd_side < best_side:
            best_side = cd_side
            best_positions = cd_pos
            best_angles = cd_ang
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
        
        # 2. Nelder-Mead Polish with tighter tolerances
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 4.0:
            def polish_objective(params):
                pos = np.zeros((N, 2))
                ang = np.zeros(N)
                for i in range(N):
                    pos[i, 0] = params[3*i]
                    pos[i, 1] = params[3*i+1]
                    ang[i] = params[3*i+2]
                    
                angles_rad = np.radians(ang)
                for i in range(N):
                    for j in range(i+1, N):
                        h1 = (pos[i, 0], pos[i, 1], angles_rad[i])
                        h2 = (pos[j, 0], pos[j, 1], angles_rad[j])
                        if not hexagons_disjoint_fast(h1, h2):
                            return 1e6
                
                side, phi, cx, cy = compute_min_outer_side(pos, ang)
                return side

            init_params = np.zeros(3*N)
            for i in range(N):
                init_params[3*i] = best_positions[i, 0]
                init_params[3*i+1] = best_positions[i, 1]
                init_params[3*i+2] = best_angles[i]
                
            try:
                res = minimize(polish_objective, init_params, method='Nelder-Mead', 
                               options={'maxiter': 8000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
                
                if res.fun < best_side:
                    new_pos = np.zeros((N, 2))
                    new_ang = np.zeros(N)
                    for i in range(N):
                        new_pos[i, 0] = res.x[3*i]
                        new_pos[i, 1] = res.x[3*i+1]
                        new_ang[i] = res.x[3*i+2]
                    
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
        
        # 3. Final high-precision coordinate descent
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 2.0:
            final_cd_time = min(remaining_time * 0.5, 8.0)
            final_side, final_pos, final_ang = polish_coordinate_descent(best_positions, best_angles, final_cd_time)
            if final_side < best_side:
                best_side = final_side
                best_positions = final_pos
                best_angles = final_ang
                best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    # Final verification
    if not is_valid_configuration(best_positions, best_angles):
        for pos_f, ang_f in inits:
            if is_valid_configuration(pos_f, ang_f):
                best_positions = pos_f.copy()
                best_angles = ang_f.copy()
                best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
                break
        else:
            best_positions, best_angles = make_honeycomb()
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
