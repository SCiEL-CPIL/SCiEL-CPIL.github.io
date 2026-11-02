import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 12
DEADLINE = 140.0  # Safety margin before 150s wall-clock kill

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
        # Vertices as (6,2) arrays
        v1 = np.column_stack((h1[0] + _ux * c1 - _uy * s1,
                              h1[1] + _ux * s1 + _uy * c1))
        v2 = np.column_stack((h2[0] + _ux * c2 - _uy * s2,
                              h2[1] + _ux * s2 + _uy * c2))
        # Edge normals rotated
        en1 = np.column_stack((_enx * c1 - _eny * s1,
                               _enx * s1 + _eny * c1))
        en2 = np.column_stack((_enx * c2 - _eny * s2,
                               _enx * s2 + _eny * c2))
        # Project both sets onto all 12 normals at once
        all_normals = np.vstack((en1, en2))  # (12, 2)
        p1 = v1 @ all_normals.T  # (6, 12)
        p2 = v2 @ all_normals.T  # (6, 12)
        p1_max = p1.max(axis=0)
        p1_min = p1.min(axis=0)
        p2_max = p2.max(axis=0)
        p2_min = p2.min(axis=0)
        # Separating axis found if any normal separates
        if np.any(p1_max < p2_min) or np.any(p2_max < p1_min):
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
        
        # Coarse scan first to find a good region
        best_side = float('inf')
        best_phi = 0.0
        for phi_deg in range(0, 60, 1):
            s = eval_phi(phi_deg)
            if s < best_side:
                best_side = s
                best_phi = phi_deg
        
        # Fine tune with minimize_scalar
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

    def smooth_outer_side(positions, angles_deg, beta=50.0):
        """Smooth surrogate using log-sum-exp for the max over projections."""
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        def eval_phi_smooth(phi_deg):
            phi = np.radians(phi_deg)
            normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            abs_proj = np.abs(proj)  # (72, 6)
            # Soft max over vertices for each normal direction
            # log(sum(exp(beta * x))) / beta approximates max
            scaled = beta * abs_proj
            max_scaled = scaled.max(axis=0)
            log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled), axis=0))
            soft_max = log_sum_exp / beta
            side_vals = (2.0 / sqrt3) * soft_max
            # Soft max over the 6 normal directions
            scaled2 = beta * side_vals
            max_scaled2 = scaled2.max()
            log_sum_exp2 = max_scaled2 + np.log(np.sum(np.exp(scaled2 - max_scaled2)))
            return log_sum_exp2 / beta
        
        # Coarse scan
        best_side = float('inf')
        best_phi = 0.0
        for phi_deg in range(0, 60, 1):
            s = eval_phi_smooth(phi_deg)
            if s < best_side:
                best_side = s
                best_phi = phi_deg
        
        # Fine tune
        try:
            lo = max(0.0, best_phi - 2.0)
            hi = min(60.0, best_phi + 2.0)
            res = minimize_scalar(eval_phi_smooth, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-6})
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

    # Honeycomb initial configuration
    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3/2.0, 1.5])
    lattice_coords = [
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, -1), (-1, 1), (2, -1), (-2, 1),
        (1, 1), (-1, -1), (2, 0),
    ]
    base_positions = np.zeros((N, 2))
    base_angles = np.zeros(N)
    for idx, (i, j) in enumerate(lattice_coords):
        base_positions[idx] = i * a1 + j * a2
        base_angles[idx] = 0.0

    best_side = float('inf')
    best_positions = base_positions.copy()
    best_angles = base_angles.copy()
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    # Strategy: Run SA for 55% of the time, then Polish for the rest
    # More time for smooth polish with log-sum-exp surrogate
    sa_time_budget = DEADLINE * 0.55
    
    # Single robust SA run with adaptive parameters
    current_positions = base_positions.copy()
    current_angles = base_angles.copy()
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
        
        dx = random.gauss(0, step_size)
        dy = random.gauss(0, step_size)
        da = random.gauss(0, 5.0)
        
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
        
        # Adaptive step size
        if total_count % 500 == 0:
            acc_rate = accept_count / total_count
            if acc_rate < 0.1:
                step_size *= 0.8
                if step_size < 0.005: step_size = 0.005
            elif acc_rate > 0.5:
                step_size *= 1.2
                if step_size > 1.0: step_size = 1.0
            accept_count = 0
            total_count = 0
            
        T *= cooling
        if T < T_min: T = T_min
        step_size *= step_decay
        if step_size < 0.005: step_size = 0.005

    # Polish Phase
    remaining_time = DEADLINE - (time.time() - t0)
    
    if remaining_time > 2.0:
        # 1. Coordinate Descent Polish (shorter, to save time for smooth polish)
        def polish_coordinate_descent(pos, ang, time_limit):
            current_side, _, _, _ = compute_min_outer_side(pos, ang)
            best_side_cd = current_side
            best_pos_cd = pos.copy()
            best_ang_cd = ang.copy()
            
            t_start = time.time()
            step = 0.05
            min_step = 0.001
            
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

        cd_time = min(remaining_time * 0.25, 10.0)
        cd_side, cd_pos, cd_ang = polish_coordinate_descent(best_positions, best_angles, cd_time)
        
        if cd_side < best_side:
            best_side = cd_side
            best_positions = cd_pos
            best_angles = cd_ang
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
        
        # 2. Smooth log-sum-exp polish using SLSQP with smooth objective
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 5.0:
            def smooth_objective(params):
                pos = params[:2*N].reshape(N, 2)
                ang = params[2*N:]
                
                # Quick validity check
                angles_rad = np.radians(ang)
                for i in range(N):
                    for j in range(i+1, N):
                        h1 = (pos[i, 0], pos[i, 1], angles_rad[i])
                        h2 = (pos[j, 0], pos[j, 1], angles_rad[j])
                        if not hexagons_disjoint_fast(h1, h2):
                            return 1e4
                
                side, phi, cx, cy = smooth_outer_side(pos, ang, beta=50.0)
                return side

            init_params = np.zeros(3*N)
            for i in range(N):
                init_params[2*i] = best_positions[i, 0]
                init_params[2*i+1] = best_positions[i, 1]
                init_params[2*N+i] = best_angles[i]
                
            try:
                # Use SLSQP for smooth optimization
                res = minimize(smooth_objective, init_params, method='SLSQP',
                               options={'maxiter': 3000, 'ftol': 1e-8, 'maxfun': 5000})
                
                if res.fun < best_side:
                    new_pos = res.x[:2*N].reshape(N, 2)
                    new_ang = res.x[2*N:]
                    
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
        
        # 3. Final Nelder-Mead with exact objective (shorter)
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 3.0:
            def polish_objective(params):
                pos = params[:2*N].reshape(N, 2)
                ang = params[2*N:]
                
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
                init_params[2*i] = best_positions[i, 0]
                init_params[2*i+1] = best_positions[i, 1]
                init_params[2*N+i] = best_angles[i]
                
            try:
                res = minimize(polish_objective, init_params, method='Nelder-Mead', 
                               options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-9})
                
                if res.fun < best_side:
                    new_pos = res.x[:2*N].reshape(N, 2)
                    new_ang = res.x[2*N:]
                    
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

    # Final verification
    if not is_valid_configuration(best_positions, best_angles):
        best_positions = base_positions.copy()
        best_angles = base_angles.copy()
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
    
    # Final high-precision phi optimization
    try:
        final_side, final_phi, final_cx, final_cy = compute_min_outer_side(best_positions, best_angles)
        if final_side < best_side:
            best_side = final_side
            best_phi = final_phi
            best_cx = final_cx
            best_cy = final_cy
    except Exception:
        pass
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
