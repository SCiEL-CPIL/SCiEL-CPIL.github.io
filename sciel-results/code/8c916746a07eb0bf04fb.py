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

    def is_valid_configuration(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        for i in range(N):
            for j in range(i+1, N):
                h1 = (positions[i][0], positions[i][1], angles_rad[i])
                h2 = (positions[j][0], positions[j][1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    return False
        return True

    # --- NEW: Force-Directed Graph Initialization ---
    def fdg_initialization():
        # Start with a random or lattice configuration
        # Lattice is safer
        a1 = np.array([sqrt3, 0.0])
        a2 = np.array([sqrt3/2.0, 1.5])
        lattice_coords = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, -1), (-1, 1), (2, -1), (-2, 1),
            (1, 1), (-1, -1), (2, 0),
        ]
        pos = np.zeros((N, 2))
        for idx, (i, j) in enumerate(lattice_coords):
            pos[idx] = i * a1 + j * a2
        
        # Relax with repulsive forces
        # We want to minimize a potential energy that penalizes overlaps and encourages compactness
        # E = sum_{i<j} (1/d_ij)^2 if d_ij < R_overlap + lambda * sum d_ij^2 (compactness)
        
        t_start = time.time()
        max_time = 2.0 # 2 seconds for FDG
        
        while time.time() - t_start < max_time:
            forces = np.zeros((N, 2))
            for i in range(N):
                for j in range(i+1, N):
                    d_vec = pos[i] - pos[j]
                    d = np.linalg.norm(d_vec)
                    if d < 1e-6:
                        d = 1e-6
                        d_vec = np.array([random.uniform(-1,1), random.uniform(-1,1)])
                        d_vec /= np.linalg.norm(d_vec)
                        d = 1e-6
                    
                    # Repulsion if close (approximating hexagon collision)
                    # Hexagons of side 1 have inradius sqrt(3)/2 ~ 0.866, circumradius 1.
                    # Minimum distance to avoid overlap is roughly 2 * inradius = 1.732 for parallel sides?
                    # Actually, for two unit hexagons, min distance is 2*sqrt(3)/2 = sqrt(3) ~ 1.732 if aligned.
                    # If rotated, it can be less. Let's use a soft repulsion starting at 2.0.
                    if d < 2.0:
                        # Strong repulsion
                        f_rep = (2.0 - d) / d
                        forces[i] += f_rep * d_vec
                        forces[j] -= f_rep * d_vec
                    
                    # Weak attraction to keep them together (compactness)
                    # This helps the gradient descent later by keeping the cluster tight
                    f_att = 0.01 * d
                    forces[i] -= f_att * d_vec / d
                    forces[j] += f_att * d_vec / d
            
            # Update positions with damping
            lr = 0.5
            pos += lr * forces
            # Optional: center of mass correction
            com = np.mean(pos, axis=0)
            pos -= com
        
        return pos

    # --- NEW: Smooth Surrogate Objective for Gradient Descent ---
    def smooth_objective(params, beta=50.0):
        """
        params: [x0, y0, a0, x1, y1, a1, ...]
        Returns: smooth outer side + collision penalty
        """
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(N):
            pos[i, 0] = params[3*i]
            pos[i, 1] = params[3*i+1]
            ang[i] = params[3*i+2]
        
        # 1. Smooth Outer Side
        angles_rad = np.radians(ang)
        all_verts = get_all_vertices(pos, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        # Optimize over phi (rotation of container)
        # For speed in gradient descent, we might fix phi to the best estimate or include it in params.
        # Including phi in params makes it 3N+1 dimensions. Let's stick to 3N for now and assume optimal phi is found in polish.
        # Actually, the "side" depends on phi. To make it differentiable, we can maximize the projection over 6 directions.
        # The max projection is non-smooth. Log-sum-exp makes it smooth.
        
        best_side = 0.0
        # We need to minimize the max projection.
        # This is equivalent to finding the minimum enclosing hexagon.
        # For a fixed set of points, the minimum enclosing hexagon side length is determined by the max projection.
        # Since we can rotate the hexagon, we want min_phi max_proj(phi).
        # This is a min-max problem. 
        # Approximation: Assume the optimal phi is close to 0 or scan a few fixed phis?
        # For gradient descent, we can just use the projection at phi=0 and let the polish step handle phi optimization.
        # Or, better: include phi in the parameter vector.
        
        # Let's modify params to include phi at the end: [x0, y0, a0, ..., x11, y11, a11, phi]
        # Total dims: 3*N + 1
        
        phi = params[-1]
        normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
        abs_proj = np.abs(proj)
        
        # Log-sum-exp for max
        max_val = np.max(abs_proj, axis=0)
        shifted = (abs_proj - max_val[None, :]) * beta
        log_sum_exp = max_val + np.log(np.sum(np.exp(shifted), axis=0)) / beta
        side_vals = (2.0 / sqrt3) * log_sum_exp
        side = np.max(side_vals)
        
        # 2. Collision Penalty
        # Smooth approximation of collision.
        # If hexagons overlap, we want a large penalty.
        # We can use the sum of squared penetration depths.
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                h1 = (pos[i, 0], pos[i, 1], angles_rad[i])
                h2 = (pos[j, 0], pos[j, 1], angles_rad[j])
                # Check for collision using the same logic as hexagons_disjoint_fast but return penetration
                # This is expensive. For gradient descent, we need a differentiable proxy.
                # Proxy: Distance between centers. If d < D_min, penalty = (D_min - d)^2.
                # D_min depends on angles. 
                # Simplified proxy: Assume worst-case alignment or average.
                # Let's use a simple radial penalty.
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                d = math.sqrt(dx*dx + dy*dy)
                # Min distance for non-overlap of two unit hexagons is approx 1.732 (sqrt(3)) for parallel sides.
                # With rotation, it can be as low as 1.5? No, inradius is 0.866. Sum is 1.732.
                # If one is rotated 30 deg, the distance can be smaller?
                # The minimum distance between two unit hexagons is actually sqrt(3) if they share a side parallel.
                # If rotated, they can get closer? No, the inradius is the minimum distance to the boundary.
                # So the minimum center distance is 2 * inradius = sqrt(3) ~ 1.732.
                # Wait, if they are rotated, the "width" changes.
                # Let's use a safe threshold of 1.7 for the penalty to trigger.
                D_min = 1.7
                if d < D_min:
                    penalty += (D_min - d) ** 2
        
        return side + 10.0 * penalty

    # --- MAIN OPTIMIZATION LOOP ---
    
    # 1. Initialize with FDG
    initial_pos = fdg_initialization()
    initial_angles = np.zeros(N)
    
    # Initial parameters for minimize: [x, y, angle] * N + [phi]
    init_params = np.zeros(3*N + 1)
    for i in range(N):
        init_params[3*i] = initial_pos[i, 0]
        init_params[3*i+1] = initial_pos[i, 1]
        init_params[3*i+2] = initial_angles[i]
    init_params[-1] = 0.0 # Initial phi
    
    best_side = float('inf')
    best_positions = initial_pos.copy()
    best_angles = initial_angles.copy()
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0
    
    # Evaluate initial
    cur_side, cur_phi, cur_cx, cur_cy = compute_min_outer_side(initial_pos, initial_angles)
    if cur_side < best_side:
        best_side = cur_side
        best_positions = initial_pos.copy()
        best_angles = initial_angles.copy()
        best_phi = cur_phi
        best_cx = cur_cx
        best_cy = cur_cy

    # 2. Gradient Descent on Smooth Surrogate
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 10.0:
        try:
            # Use L-BFGS-B
            # We need to ensure the function is differentiable. 
            # The log-sum-exp is smooth. The penalty (D_min - d)^2 is smooth except at d=D_min.
            # We can smooth the penalty with a softplus or similar, but (D_min-d)^2 is fine if we start far away.
            
            res = minimize(smooth_objective, init_params, method='L-BFGS-B', 
                           options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-10, 'maxfun': 1000})
            
            # Extract results
            new_pos = np.zeros((N, 2))
            new_ang = np.zeros(N)
            new_phi = res.x[-1]
            for i in range(N):
                new_pos[i, 0] = res.x[3*i]
                new_pos[i, 1] = res.x[3*i+1]
                new_ang[i] = res.x[3*i+2]
            
            # Check validity
            if is_valid_configuration(new_pos, new_ang):
                new_side, new_phi_opt, new_cx, new_cy = compute_min_outer_side(new_pos, new_ang)
                if new_side < best_side:
                    best_side = new_side
                    best_positions = new_pos
                    best_angles = new_ang
                    best_phi = new_phi_opt
                    best_cx = new_cx
                    best_cy = new_cy
            else:
                # If invalid, try to repair with a few steps of SA or CD
                # For now, just ignore if invalid, relying on the initial valid state or polish.
                # But we can try to nudge it back.
                pass
                
        except Exception as e:
            # Fallback to SA if GD fails
            pass

    # 3. Polish Phase (Coordinate Descent)
    # This is crucial to fix any small overlaps and optimize phi precisely.
    
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 5.0:
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

        cd_time = min(remaining_time * 0.5, 20.0)
        cd_side, cd_pos, cd_ang = polish_coordinate_descent(best_positions, best_angles, cd_time)
        
        if cd_side < best_side:
            best_side = cd_side
            best_positions = cd_pos
            best_angles = cd_ang
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
            
    # Final verification
    if not is_valid_configuration(best_positions, best_angles):
        # Fallback to initial FDG result if it was valid
        if is_valid_configuration(initial_pos, initial_angles):
            best_positions = initial_pos.copy()
            best_angles = initial_angles.copy()
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
        else:
            # Last resort: Lattice
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
            best_positions = base_positions
            best_angles = base_angles
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
