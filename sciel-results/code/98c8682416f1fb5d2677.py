import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 12
DEADLINE = 135.0  # Safety margin before 150s wall-clock kill

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

    def check_overlap_single(moved_idx, positions, angles_rad):
        """Check if hexagon at moved_idx overlaps with any other. Returns True if no overlap."""
        h1x = positions[moved_idx, 0]
        h1y = positions[moved_idx, 1]
        a1 = angles_rad[moved_idx]
        c1, s1 = math.cos(a1), math.sin(a1)
        v1x = h1x + _ux * c1 - _uy * s1
        v1y = h1y + _ux * s1 + _uy * c1
        en1x = _enx * c1 - _eny * s1
        en1y = _enx * s1 + _eny * c1
        
        for j in range(N):
            if j == moved_idx:
                continue
            dx = h1x - positions[j, 0]
            dy = h1y - positions[j, 1]
            dist_sq = dx*dx + dy*dy
            if dist_sq > 4.5:
                continue
            a2 = angles_rad[j]
            c2, s2 = math.cos(a2), math.sin(a2)
            v2x = positions[j, 0] + _ux * c2 - _uy * s2
            v2y = positions[j, 1] + _ux * s2 + _uy * c2
            en2x = _enx * c2 - _eny * s2
            en2y = _enx * s2 + _eny * c2
            for i in range(6):
                nx, ny = en1x[i], en1y[i]
                p1 = v1x * nx + v1y * ny
                p2 = v2x * nx + v2y * ny
                if p1.max() < p2.min() or p2.max() < p1.min():
                    break
            else:
                for i in range(6):
                    nx, ny = en2x[i], en2y[i]
                    p1 = v1x * nx + v1y * ny
                    p2 = v2x * nx + v2y * ny
                    if p1.max() < p2.min() or p2.max() < p1.min():
                        break
                else:
                    return False
        return True

    def compute_min_outer_side_fast(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        # Vectorized phi evaluation
        best_side = float('inf')
        best_phi = 0.0
        
        # Coarse scan
        for phi_deg in range(0, 60, 2):
            phi = np.radians(phi_deg)
            normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            abs_proj = np.abs(proj)
            max_projs = np.max(abs_proj, axis=0)
            side_vals = (2.0 / sqrt3) * max_projs
            s = np.max(side_vals)
            if s < best_side:
                best_side = s
                best_phi = phi_deg
        
        # Fine tune
        try:
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
            res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-7})
            if res.success and res.fun < best_side:
                best_side = res.fun
                best_phi = res.x
        except Exception:
            pass
            
        return best_side, best_phi, center[0], center[1]

    def is_valid_configuration(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        for i in range(N):
            if not check_overlap_single(i, positions, angles_rad):
                return False
        return True

    # Try multiple initial configurations with diverse structures
    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3/2.0, 1.5])
    
    # Config 1: Standard honeycomb cluster
    config1_coords = [
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, -1), (-1, 1), (2, -1), (-2, 1),
        (1, 1), (-1, -1), (2, 0),
    ]
    
    # Config 2: Alternative honeycomb
    config2_coords = [
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, -1), (-1, 1), (2, -1), (-2, 1),
        (2, 1), (-2, -1), (0, 2),
    ]
    
    # Config 3: Linear arrangement with offsets
    config3_coords = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1),
    ]
    
    # Config 4: Two rows staggered
    config4_coords = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0),
        (0, -1), (1, -1), (2, -1), (3, -1), (4, -1), (5, -1),
    ]
    
    # Config 5: Compact 3x4 grid-like
    config5_coords = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, -1), (1, -1), (2, -1), (3, -1),
    ]
    
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0
    
    all_configs = [config1_coords, config2_coords, config3_coords, config4_coords, config5_coords]
    
    for coords in all_configs:
        base_positions = np.zeros((N, 2))
        base_angles = np.zeros(N)
        for idx, (i, j) in enumerate(coords):
            base_positions[idx] = i * a1 + j * a2
            base_angles[idx] = 0.0
        
        side, phi, cx, cy = compute_min_outer_side_fast(base_positions, base_angles)
        if side < best_side:
            best_side = side
            best_positions = base_positions.copy()
            best_angles = base_angles.copy()
            best_phi = phi
            best_cx = cx
            best_cy = cy

    # Strategy: Run SA for 75% of the time, then Polish for the rest
    sa_time_budget = DEADLINE * 0.75
    
    # SA with adaptive parameters and group moves
    current_positions = best_positions.copy()
    current_angles = best_angles.copy()
    current_side, current_phi, current_cx, current_cy = compute_min_outer_side_fast(current_positions, current_angles)
    
    if current_side < best_side:
        best_side = current_side
        best_positions = current_positions.copy()
        best_angles = current_angles.copy()
        best_phi = current_phi
        best_cx = current_cx
        best_cy = current_cy

    T = 0.3
    T_min = 0.00005
    cooling = 0.99993
    step_size = 0.3
    step_decay = 0.99999
    
    step_count = 0
    max_steps = 500000
    accept_count = 0
    total_count = 0
    no_improve_count = 0
    last_improve_step = 0
    
    while time.time() - t0 < sa_time_budget and step_count < max_steps:
        step_count += 1
        total_count += 1
        
        # Occasionally do a 2-hexagon move to escape local minima
        if random.random() < 0.15:
            # Group move: pick 2 distinct hexagons
            idxs = random.sample(range(N), 2)
            idx1, idx2 = idxs[0], idxs[1]
            new_positions = current_positions.copy()
            new_angles = current_angles.copy()
            
            for idx in [idx1, idx2]:
                dx = random.gauss(0, step_size)
                dy = random.gauss(0, step_size)
                da = random.gauss(0, 5.0)
                new_positions[idx, 0] += dx
                new_positions[idx, 1] += dy
                new_angles[idx] += da
            
            angles_rad_new = np.radians(new_angles)
            valid = check_overlap_single(idx1, new_positions, angles_rad_new)
            if valid:
                valid = check_overlap_single(idx2, new_positions, angles_rad_new)
            
            if not valid:
                T *= cooling
                if T < T_min: T = T_min
                continue
        else:
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
            valid = check_overlap_single(idx, new_positions, angles_rad_new)
            
            if not valid:
                T *= cooling
                if T < T_min: T = T_min
                continue
        
        new_side, new_phi, new_cx, new_cy = compute_min_outer_side_fast(new_positions, new_angles)
        
        improved = False
        if new_side < current_side:
            current_positions = new_positions
            current_angles = new_angles
            current_side = new_side
            current_phi = new_phi
            current_cx = new_cx
            current_cy = new_cy
            accept_count += 1
            improved = True
            if new_side < best_side:
                best_side = new_side
                best_positions = new_positions.copy()
                best_angles = new_angles.copy()
                best_phi = new_phi
                best_cx = new_cx
                best_cy = new_cy
                no_improve_count = 0
                last_improve_step = step_count
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
        
        if not improved:
            no_improve_count += 1
        else:
            no_improve_count = 0
        
        # Adaptive step size and temperature
        if total_count % 200 == 0:
            acc_rate = accept_count / total_count
            if acc_rate < 0.05:
                step_size *= 0.6
                if step_size < 0.002: step_size = 0.002
            elif acc_rate > 0.5:
                step_size *= 1.4
                if step_size > 1.0: step_size = 1.0
            accept_count = 0
            total_count = 0
            
        T *= cooling
        if T < T_min: T = T_min
        step_size *= step_decay
        if step_size < 0.002: step_size = 0.002

    # Polish Phase
    remaining_time = DEADLINE - (time.time() - t0)
    
    if remaining_time > 2.0:
        # 1. Coordinate Descent Polish with adaptive steps
        def polish_coordinate_descent(pos, ang, time_limit):
            current_side, _, _, _ = compute_min_outer_side_fast(pos, ang)
            best_side_cd = current_side
            best_pos_cd = pos.copy()
            best_ang_cd = ang.copy()
            
            t_start = time.time()
            step = 0.05
            min_step = 0.0003
            
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
                            valid = check_overlap_single(idx, new_pos, angles_rad_new)
                            
                            if not valid:
                                continue
                            
                            new_side, _, _, _ = compute_min_outer_side_fast(new_pos, new_ang)
                            
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
                    step *= 0.7
                    continue
                break
            
            return best_side_cd, best_pos_cd, best_ang_cd

        cd_time = min(remaining_time * 0.4, 25.0)
        cd_side, cd_pos, cd_ang = polish_coordinate_descent(best_positions, best_angles, cd_time)
        
        if cd_side < best_side:
            best_side = cd_side
            best_positions = cd_pos
            best_angles = cd_ang
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side_fast(best_positions, best_angles)
        
        # 2. Nelder-Mead Polish with more iterations
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 5.0:
            def polish_objective(params):
                pos = np.zeros((N, 2))
                ang = np.zeros(N)
                for i in range(N):
                    pos[i, 0] = params[3*i]
                    pos[i, 1] = params[3*i+1]
                    ang[i] = params[3*i+2]
                    
                angles_rad = np.radians(ang)
                for i in range(N):
                    if not check_overlap_single(i, pos, angles_rad):
                        return 1e6
                
                side, phi, cx, cy = compute_min_outer_side_fast(pos, ang)
                return side

            init_params = np.zeros(3*N)
            for i in range(N):
                init_params[3*i] = best_positions[i, 0]
                init_params[3*i+1] = best_positions[i, 1]
                init_params[3*i+2] = best_angles[i]
                
            try:
                res = minimize(polish_objective, init_params, method='Nelder-Mead', 
                               options={'maxiter': 15000, 'xatol': 1e-8, 'fatol': 1e-10,
                                        'initial_simplex': 0.02})
                
                if res.fun < best_side:
                    new_pos = np.zeros((N, 2))
                    new_ang = np.zeros(N)
                    for i in range(N):
                        new_pos[i, 0] = res.x[3*i]
                        new_pos[i, 1] = res.x[3*i+1]
                        new_ang[i] = res.x[3*i+2]
                    
                    if is_valid_configuration(new_pos, new_ang):
                        new_side, new_phi, new_cx, new_cy = compute_min_outer_side_fast(new_pos, new_ang)
                        if new_side < best_side:
                            best_side = new_side
                            best_positions = new_pos
                            best_angles = new_ang
                            best_phi = new_phi
                            best_cx = new_cx
                            best_cy = new_cy
            except Exception:
                pass

    # Final verification
    if not is_valid_configuration(best_positions, best_angles):
        # Fallback to a valid configuration
        a1_fb = np.array([sqrt3, 0.0])
        a2_fb = np.array([sqrt3/2.0, 1.5])
        fb_coords = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, -1), (-1, 1), (2, -1), (-2, 1),
            (1, 1), (-1, -1), (2, 0),
        ]
        best_positions = np.zeros((N, 2))
        best_angles = np.zeros(N)
        for idx, (i, j) in enumerate(fb_coords):
            best_positions[idx] = i * a1_fb + j * a2_fb
            best_angles[idx] = 0.0
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side_fast(best_positions, best_angles)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
