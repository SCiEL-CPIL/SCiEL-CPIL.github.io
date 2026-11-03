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
        # Vertices (6,)
        v1x = h1[0] + _ux * c1 - _uy * s1
        v1y = h1[1] + _ux * s1 + _uy * c1
        v2x = h2[0] + _ux * c2 - _uy * s2
        v2y = h2[1] + _ux * s2 + _uy * c2
        # Edge normals (6,)
        en1x = _enx * c1 - _eny * s1
        en1y = _enx * s1 + _eny * c1
        en2x = _enx * c2 - _eny * s2
        en2y = _enx * s2 + _eny * c2
        # Project h1 vertices onto h1's edge normals (6 axes)
        # p1_proj[i] = v1x[i]*en1x[i] + v1y[i]*en1y[i]  -- but SAT uses each normal for ALL vertices
        # For axis i of hex1: project all 6 vertices of both hexes
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
        
        def eval_phi_smooth(phi_deg, beta=50.0):
            """Log-sum-exp smooth approximation of the max side length."""
            phi = np.radians(phi_deg)
            normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            abs_proj = np.abs(proj)
            side_vals = (2.0 / sqrt3) * abs_proj  # shape (72, 6)
            # Smooth max over all 72*6 values via log-sum-exp
            flat_sv = side_vals.ravel()
            m = flat_sv.max()
            if m < 1e-12:
                return 0.0
            log_sum = m + np.log(np.sum(np.exp(beta * (flat_sv - m)))) / beta
            return log_sum
        
        # Coarse scan first to find a good region
        best_side = float('inf')
        best_phi = 0.0
        for phi_deg in range(0, 60, 1):
            s = eval_phi(phi_deg)
            if s < best_side:
                best_side = s
                best_phi = phi_deg
        
        # Fine tune with minimize_scalar on the exact (nonsmooth) objective
        try:
            lo = max(0.0, best_phi - 1.5)
            hi = min(60.0, best_phi + 1.5)
            res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-10, 'maxiter': 200})
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
    # More polish time since smooth surrogate enables faster convergence
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

    T = 0.4
    T_min = 0.0001
    cooling = 0.99995
    step_size = 0.25
    step_decay = 0.99998
    
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
        if total_count % 300 == 0:
            acc_rate = accept_count / total_count
            if acc_rate < 0.08:
                step_size *= 0.75
                if step_size < 0.003: step_size = 0.003
            elif acc_rate > 0.45:
                step_size *= 1.25
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
        # 1. Smooth surrogate-based optimization using L-BFGS-B
        #    Uses log-sum-exp to make the outer-side objective differentiable
        def smooth_outer_side(pos, ang):
            """Differentiable approximation of outer side length using log-sum-exp."""
            angles_rad = np.radians(ang)
            all_verts = get_all_vertices(pos, angles_rad)
            verts_flat = all_verts.reshape(-1, 2)
            center = np.mean(verts_flat, axis=0)
            offsets = verts_flat - center
            
            # For a given phi, compute smooth max side
            # We optimize over phi as well, but for gradient w.r.t. positions/angles,
            # we fix phi at the best known value
            # Actually, let's compute the smooth objective that includes phi optimization
            # by scanning a small range and taking the smooth max
            # Simpler: use the exact phi optimization but with smooth inner max
            
            # Use beta for log-sum-exp over the 72 vertices per axis
            beta = 30.0
            
            def smooth_side_for_phi(phi_rad):
                normal_angles = phi_rad + np.pi/6 + np.arange(6) * np.pi/3
                nx = np.cos(normal_angles)
                ny = np.sin(normal_angles)
                proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
                abs_proj = np.abs(proj)
                side_vals = (2.0 / sqrt3) * abs_proj  # (72, 6)
                flat = side_vals.ravel()
                m = flat.max()
                if m < 1e-12:
                    return 0.0
                return m + np.log(np.sum(np.exp(beta * (flat - m)))) / beta
            
            # Coarse scan for phi
            best_s = float('inf')
            best_p = 0.0
            for p_deg in range(0, 60, 2):
                s = smooth_side_for_phi(np.radians(p_deg))
                if s < best_s:
                    best_s = s
                    best_p = np.radians(p_deg)
            
            # Refine phi with a few golden-section steps
            for _ in range(8):
                lo_p = max(0.0, best_p - 0.1)
                hi_p = min(np.pi/3, best_p + 0.1)
                gr = (1 + math.sqrt(5)) / 2
                x1 = hi_p - (hi_p - lo_p) / gr
                x2 = lo_p + (hi_p - lo_p) / gr
                f1 = smooth_side_for_phi(x1)
                f2 = smooth_side_for_phi(x2)
                if f1 < f2:
                    hi_p = x2
                else:
                    lo_p = x1
                best_p = (lo_p + hi_p) / 2
                best_s = smooth_side_for_phi(best_p)
            
            return best_s, best_p, center
        
        # Objective for L-BFGS-B: smooth side + collision penalty
        def smooth_objective(params):
            pos = np.zeros((N, 2))
            ang = np.zeros(N)
            for i in range(N):
                pos[i, 0] = params[3*i]
                pos[i, 1] = params[3*i+1]
                ang[i] = params[3*i+2]
            
            # Collision penalty: sum of squared overlaps (soft)
            angles_rad = np.radians(ang)
            penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    h1 = (pos[i,0], pos[i,1], angles_rad[i])
                    h2 = (pos[j,0], pos[j,1], angles_rad[j])
                    if not hexagons_disjoint_fast(h1, h2):
                        # Compute penetration depth as penalty
                        # Use center distance as proxy
                        dx = pos[i,0] - pos[j,0]
                        dy = pos[i,1] - pos[j,1]
                        d = math.sqrt(dx*dx + dy*dy)
                        # Minimum distance for disjoint unit hexagons is ~1.732 (sqrt(3))
                        overlap = max(0, 1.732 - d)
                        penalty += overlap * overlap
            if penalty > 0:
                return 1e6 + penalty
            
            s, _, _ = smooth_outer_side(pos, ang)
            return s
        
        init_params = np.zeros(3*N)
        for i in range(N):
            init_params[3*i] = best_positions[i, 0]
            init_params[3*i+1] = best_positions[i, 1]
            init_params[3*i+2] = best_angles[i]
        
        lbfgs_time = min(remaining_time * 0.5, 20.0)
        t_lbfgs_start = time.time()
        
        try:
            res = minimize(smooth_objective, init_params, method='L-BFGS-B',
                           options={'maxiter': 3000, 'ftol': 1e-10, 'gtol': 1e-8,
                                    'maxfun': 50000})
            
            if res.fun < best_side + 0.01:
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
        
        # 2. Coordinate Descent Polish (tighter)
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 3.0:
            def polish_coordinate_descent(pos, ang, time_limit):
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

            cd_time = min(remaining_time * 0.5, 15.0)
            cd_side, cd_pos, cd_ang = polish_coordinate_descent(best_positions, best_angles, cd_time)
            
            if cd_side < best_side:
                best_side = cd_side
                best_positions = cd_pos
                best_angles = cd_ang
                best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
        
        # 3. Final Nelder-Mead polish with tight tolerances
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 3.0:
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
                               options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10})
                
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

    # Final verification and high-precision phi optimization
    if not is_valid_configuration(best_positions, best_angles):
        best_positions = base_positions.copy()
        best_angles = base_angles.copy()
    
    # Final high-precision outer hexagon optimization
    angles_rad_final = np.radians(best_angles)
    all_verts_final = get_all_vertices(best_positions, angles_rad_final)
    verts_flat_final = all_verts_final.reshape(-1, 2)
    center_final = np.mean(verts_flat_final, axis=0)
    offsets_final = verts_flat_final - center_final
    
    def final_eval_phi(phi_deg):
        phi = np.radians(phi_deg)
        normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets_final[:, 0:1] * nx[None, :] + offsets_final[:, 1:2] * ny[None, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / sqrt3) * max_projs
        return np.max(side_vals)
    
    # Multi-start fine phi search
    best_final_side = float('inf')
    best_final_phi = 0.0
    for start in range(0, 60, 3):
        try:
            res = minimize_scalar(final_eval_phi, bounds=(start, start + 3), method='bounded',
                                  options={'xatol': 1e-12, 'maxiter': 500})
            if res.success and res.fun < best_final_side:
                best_final_side = res.fun
                best_final_phi = res.x
        except Exception:
            pass
    
    best_side = best_final_side
    best_phi = best_final_phi
    best_cx = center_final[0]
    best_cy = center_final[1]
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
