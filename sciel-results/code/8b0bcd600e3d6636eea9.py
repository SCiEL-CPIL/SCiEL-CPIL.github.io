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
        
        best_side = float('inf')
        best_phi = 0.0
        for phi_deg in range(0, 60, 1):
            s = eval_phi(phi_deg)
            if s < best_side:
                best_side = s
                best_phi = phi_deg
                
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

    def is_valid_configuration(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        for i in range(N):
            for j in range(i+1, N):
                h1 = (positions[i][0], positions[i][1], angles_rad[i])
                h2 = (positions[j][0], positions[j][1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    return False
        return True

    # Smooth Surrogate Objective for L-BFGS-B
    def smooth_objective(params):
        # 1. Unpack
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(N):
            pos[i, 0] = params[3*i]
            pos[i, 1] = params[3*i+1]
            ang[i] = params[3*i+2]
            
        # 2. Compute outer side (smooth part)
        angles_rad = np.radians(ang)
        all_verts = get_all_vertices(pos, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        # We optimize over phi as well? No, phi is part of the outer hexagon.
        # The task allows the outer hexagon to rotate.
        # However, L-BFGS-B optimizes params. We can add phi to params?
        # Or we can fix phi to 0 and let the inner hexagons adapt?
        # The outer hexagon rotation is a degree of freedom.
        # Let's add phi to the params vector.
        # params: 3*N + 1 (phi)
        
        phi = params[-1]
        normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / sqrt3) * max_projs
        
        # Log-sum-exp for smooth max
        beta = 20.0
        m = np.max(side_vals)
        smooth_side = m + np.log(np.sum(np.exp(beta * (side_vals - m)))) / beta
        
        # 3. Compute collision penalty (smooth part)
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                h1 = (pos[i, 0], pos[i, 1], angles_rad[i])
                h2 = (pos[j, 0], pos[j, 1], angles_rad[j])
                
                # Fast distance check
                dx = h1[0] - h2[0]
                dy = h1[1] - h2[1]
                dist_sq = dx*dx + dy*dy
                if dist_sq > 4.5:
                    continue
                
                c1, s1 = math.cos(h1[2]), math.sin(h1[2])
                c2, s2 = math.cos(h2[2]), math.sin(h2[2])
                v1x = h1[0] + _ux * c1 - _uy * s1
                v1y = h1[1] + _ux * s1 + _uy * c1
                v2x = h2[0] + _ux * c2 - _uy * s2
                v2y = h2[1] + _ux * s2 + _uy * c2
                en1x = _enx * c1 - _eny * s1
                en1y = _enx * s1 + _eny * c1
                en2x = _enx * c2 - _eny * s2
                en2y = _enx * s2 + _eny * c2
                
                # Compute penetration depth (negative of separation)
                # We want to penalize overlap.
                # Separation is min over all edge normals of (max proj 1 - min proj 2) and (max proj 2 - min proj 1)
                # If they are disjoint, separation > 0.
                # If they overlap, separation < 0 (penetration).
                # We want to maximize separation, or minimize -separation (penetration).
                
                min_sep = float('inf')
                for i_e in range(6):
                    nx, ny = en1x[i_e], en1y[i_e]
                    p1 = v1x * nx + v1y * ny
                    p2 = v2x * nx + v2y * ny
                    sep = p1.max() - p2.min()
                    if sep < min_sep: min_sep = sep
                for i_e in range(6):
                    nx, ny = en2x[i_e], en2y[i_e]
                    p1 = v1x * nx + v1y * ny
                    p2 = v2x * nx + v2y * ny
                    sep = p2.max() - p1.min()
                    if sep < min_sep: min_sep = sep
                    
                # min_sep is the separation. If > 0, disjoint.
                # We want to push them apart.
                # Penalty should be large if min_sep < 0.
                # Use smooth hinge: max(0, -min_sep)
                # Smooth approximation: log(1 + exp(beta * (-min_sep))) / beta
                # If min_sep is large positive, -min_sep is large negative, exp is 0, log(1)=0.
                # If min_sep is large negative, -min_sep is large positive, exp is large, log is large.
                # This is a smooth approximation of the penetration depth.
                
                # To avoid numerical issues, clip min_sep
                if min_sep > 1.0:
                    continue
                if min_sep < -1.0:
                    penalty += 1.0 # Cap the penalty contribution per pair? No, let it grow.
                    # Actually, if min_sep is very negative, exp(beta * -min_sep) overflows.
                    # We can just use a large constant penalty if they are deeply overlapping.
                    # Or use a softer penalty.
                    # Let's use: penalty += 10.0 * max(0, -min_sep) ** 2 ?
                    # No, L-BFGS needs smooth gradients.
                    # Let's stick to log-sum-exp style but careful.
                    # If min_sep < -10, penalty is huge.
                    # Let's just add a large constant if they are "clearly" overlapping to guide away.
                    # But we want gradient.
                    # Let's use: -min_sep if min_sep < 0 else 0.
                    # Smoothed: log(1 + exp(20 * -min_sep)) / 20
                    # If min_sep = -1, -min_sep = 1. exp(20) is huge.
                    # This is effectively a hard constraint.
                    # Let's use a smaller beta for the penalty to keep gradients finite.
                    beta_pen = 5.0
                    pen_val = np.log1p(np.exp(beta_pen * (-min_sep))) / beta_pen
                    penalty += pen_val
        
        return smooth_side + 100.0 * penalty

    # Initial Configuration: Honeycomb
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
        
    # Initial params: 3*N + 1
    init_params = np.zeros(3*N + 1)
    for i in range(N):
        init_params[3*i] = base_positions[i, 0]
        init_params[3*i+1] = base_positions[i, 1]
        init_params[3*i+2] = base_angles[i]
    init_params[-1] = 0.0 # Initial phi
    
    best_side = float('inf')
    best_positions = base_positions.copy()
    best_angles = base_angles.copy()
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0
    
    # Update best with initial
    init_side, init_phi, init_cx, init_cy = compute_min_outer_side(base_positions, base_angles)
    if init_side < best_side:
        best_side = init_side
        best_positions = base_positions.copy()
        best_angles = base_angles.copy()
        best_phi = init_phi
        best_cx = init_cx
        best_cy = init_cy

    # Multi-Start L-BFGS-B
    # We will run several optimizations from slightly perturbed starts.
    
    time_limit = DEADLINE - 10.0 # Leave 10s for final polish/verification
    
    for restart in range(10):
        if time.time() - t0 > time_limit:
            break
            
        # Perturb initial params
        current_params = init_params.copy()
        if restart > 0:
            # Small random perturbation
            noise = np.random.normal(0, 0.2, 3*N)
            current_params[:3*N] += noise
            # Keep phi small
            current_params[-1] = np.random.uniform(-5, 5)
            
        try:
            res = minimize(smooth_objective, current_params, method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-5, 'maxfun': 500})
            
            if res.fun < 100.0: # Check if penalty didn't explode
                # Extract results
                new_pos = np.zeros((N, 2))
                new_ang = np.zeros(N)
                for i in range(N):
                    new_pos[i, 0] = res.x[3*i]
                    new_pos[i, 1] = res.x[3*i+1]
                    new_ang[i] = res.x[3*i+2]
                new_phi = res.x[-1]
                
                # Verify validity
                if is_valid_configuration(new_pos, new_ang):
                    new_side, new_phi_opt, new_cx, new_cy = compute_min_outer_side(new_pos, new_ang)
                    if new_side < best_side:
                        best_side = new_side
                        best_positions = new_pos.copy()
                        best_angles = new_ang.copy()
                        best_phi = new_phi_opt
                        best_cx = new_cx
                        best_cy = new_cy
        except Exception:
            pass

    # Final Polish: Coordinate Descent on the best solution found
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 2.0:
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

        cd_time = min(remaining_time * 0.8, 10.0)
        cd_side, cd_pos, cd_ang = polish_coordinate_descent(best_positions, best_angles, cd_time)
        
        if cd_side < best_side:
            best_side = cd_side
            best_positions = cd_pos
            best_angles = cd_ang
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    # Final verification
    if not is_valid_configuration(best_positions, best_angles):
        best_positions = base_positions.copy()
        best_angles = base_angles.copy()
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
