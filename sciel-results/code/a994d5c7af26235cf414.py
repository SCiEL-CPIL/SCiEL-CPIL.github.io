import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 12
DEADLINE = 115.0  # Safety margin before 120s wall-clock kill

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
        ox = offsets[:, 0]
        oy = offsets[:, 1]
        
        # Fast evaluation: only 6 critical directions per phi, vectorized
        # The outer hexagon side for direction phi is determined by max projection
        # We scan phi in 1-degree steps then refine
        
        best_side = float('inf')
        best_phi = 0.0
        
        # Precompute the 6 normal angle offsets
        normal_offsets = np.pi/6 + np.arange(6) * np.pi/3
        
        # Vectorized scan over phi values
        phi_vals = np.arange(0, 60, 1.0)
        phi_radians = np.radians(phi_vals)
        
        # For each phi, compute all 6 normal directions
        # normal_angles[phi_idx, normal_idx]
        all_normals = phi_radians[:, None] + normal_offsets[None, :]
        nx_all = np.cos(all_normals)  # shape (60, 6)
        ny_all = np.sin(all_normals)  # shape (60, 6)
        
        # proj[phi_idx, vert_idx, normal_idx] = ox[vert_idx]*nx + oy[vert_idx]*ny
        # Use broadcasting: ox[None, :, None] * nx_all[:, None, :] + oy[None, :, None] * ny_all[:, None, :]
        proj = ox[None, :, None] * nx_all[:, None, :] + oy[None, :, None] * ny_all[:, None, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=1)  # shape (60, 6)
        side_vals = (2.0 / sqrt3) * np.max(max_projs, axis=1)  # shape (60,)
        
        best_idx = np.argmin(side_vals)
        best_side = side_vals[best_idx]
        best_phi = phi_vals[best_idx]
        
        # Fine tune with minimize_scalar in a small window
        try:
            lo = max(0.0, best_phi - 1.5)
            hi = min(60.0, best_phi + 1.5)
            
            def eval_phi_fast(phi_deg):
                phi = np.radians(phi_deg)
                normal_angles = phi + normal_offsets
                nx = np.cos(normal_angles)
                ny = np.sin(normal_angles)
                proj = ox[:, None] * nx[None, :] + oy[:, None] * ny[None, :]
                abs_proj = np.abs(proj)
                max_projs = np.max(abs_proj, axis=0)
                side_vals = (2.0 / sqrt3) * max_projs
                return np.max(side_vals)
            
            res = minimize_scalar(eval_phi_fast, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-8, 'maxiter': 30})
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
                h1 = (positions[i, 0], positions[i, 1], angles_rad[i])
                h2 = (positions[j, 0], positions[j, 1], angles_rad[j])
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

    def run_sa(initial_positions, initial_angles, time_budget):
        """
        Run Simulated Annealing starting from the given configuration.
        Returns the best (side, positions, angles, phi, cx, cy) found.
        """
        current_positions = initial_positions.copy()
        current_angles = initial_angles.copy()
        current_side, current_phi, current_cx, current_cy = compute_min_outer_side(current_positions, current_angles)
        
        run_best_side = current_side
        run_best_positions = current_positions.copy()
        run_best_angles = current_angles.copy()
        run_best_phi = current_phi
        run_best_cx = current_cx
        run_best_cy = current_cy

        T = 0.2
        T_min = 0.0001
        cooling = 0.99995
        step_size = 0.2
        step_decay = 0.99999
        
        step_count = 0
        max_steps = 1000000
        accept_count = 0
        total_count = 0
        no_improve_count = 0
        improve_threshold = 500
        
        t_start = time.time()
        
        while time.time() - t_start < time_budget:
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
                no_improve_count = 0
                if new_side < run_best_side:
                    run_best_side = new_side
                    run_best_positions = new_positions.copy()
                    run_best_angles = new_angles.copy()
                    run_best_phi = new_phi
                    run_best_cx = new_cx
                    run_best_cy = new_cy
            else:
                delta = new_side - current_side
                if delta < 0: delta = 0
                if random.random() < math.exp(-delta / (T * 1.0)):
                    current_positions = new_positions
                    current_angles = new_angles
                    current_side = new_side
                    current_phi = new_phi
                    current_cx = new_cx
                    current_cy = new_cy
                    accept_count += 1
                no_improve_count += 1
            
            # Adaptive step size
            if total_count % 100 == 0:
                acc_rate = accept_count / max(total_count, 1)
                if acc_rate < 0.05:
                    step_size *= 0.7
                    if step_size < 0.005: step_size = 0.005
                elif acc_rate > 0.4:
                    step_size *= 1.3
                    if step_size > 0.5: step_size = 0.5
                accept_count = 0
                total_count = 0
            
            # Periodic restart from best if stuck
            if no_improve_count > improve_threshold:
                current_positions = run_best_positions.copy()
                current_angles = run_best_angles.copy()
                for i in range(N):
                    current_positions[i, 0] += random.gauss(0, 0.1)
                    current_positions[i, 1] += random.gauss(0, 0.1)
                    current_angles[i] += random.gauss(0, 2.0)
                # Check validity of restart, if invalid, revert to best
                angles_rad_check = np.radians(current_angles)
                valid_restart = True
                for i in range(N):
                    for j in range(i+1, N):
                        h1 = (current_positions[i, 0], current_positions[i, 1], angles_rad_check[i])
                        h2 = (current_positions[j, 0], current_positions[j, 1], angles_rad_check[j])
                        if not hexagons_disjoint_fast(h1, h2):
                            valid_restart = False
                            break
                    if not valid_restart: break
                if valid_restart:
                    current_side, current_phi, current_cx, current_cy = compute_min_outer_side(current_positions, current_angles)
                else:
                    current_positions = run_best_positions.copy()
                    current_angles = run_best_angles.copy()
                    current_side = run_best_side
                    current_phi = run_best_phi
                    current_cx = run_best_cx
                    current_cy = run_best_cy
                no_improve_count = 0
                T = max(T, 0.1)
                step_size = max(step_size, 0.15)
            
            T *= cooling
            if T < T_min: T = T_min
            step_size *= step_decay
            if step_size < 0.005: step_size = 0.005

        return run_best_side, run_best_positions, run_best_angles, run_best_phi, run_best_cx, run_best_cy

    # Generate diverse initial configurations
    def generate_starts():
        starts = []
        # 1. Base honeycomb
        starts.append((base_positions.copy(), base_angles.copy()))
        
        # 2. Rotated honeycomb (30 deg)
        rot_pos = base_positions.copy()
        rot_ang = np.zeros(N)
        for i in range(N):
            rot_ang[i] = 30.0
            # Rotate positions around center (0,0) by 30 deg? No, just rotate the hexagons.
            # Actually, rotating the whole cluster might help.
            # Let's just use different angles for the hexagons.
        starts.append((rot_pos, rot_ang))
        
        # 3. Perturbed honeycomb
        pert_pos = base_positions.copy()
        pert_ang = np.zeros(N)
        for i in range(N):
            pert_pos[i, 0] += random.gauss(0, 0.2)
            pert_pos[i, 1] += random.gauss(0, 0.2)
            pert_ang[i] = random.uniform(-10, 10)
        # Validate and keep if valid
        if is_valid_configuration(pert_pos, pert_ang):
            starts.append((pert_pos, pert_ang))
        else:
            starts.append((base_positions.copy(), base_angles.copy()))
            
        # 4. Another perturbation
        pert_pos2 = base_positions.copy()
        pert_ang2 = np.zeros(N)
        for i in range(N):
            pert_pos2[i, 0] += random.gauss(0, 0.3)
            pert_pos2[i, 1] += random.gauss(0, 0.3)
            pert_ang2[i] = random.uniform(-15, 15)
        if is_valid_configuration(pert_pos2, pert_ang2):
            starts.append((pert_pos2, pert_ang2))
        else:
            starts.append((base_positions.copy(), base_angles.copy()))
            
        return starts

    starts = generate_starts()
    
    # Determine time budget per start
    # We have ~115s. 
    # Let's allocate time to the first 3 starts.
    # Start 0: 40%
    # Start 1: 30%
    # Start 2: 20%
    # Start 3: 10%
    # Plus polish time.
    
    total_sa_time = DEADLINE * 0.8
    polish_time = DEADLINE * 0.2
    
    # We will run SA for each start, updating the global best.
    # To save time, we can run them sequentially and stop if time is running out.
    
    time_allocation = [0.4, 0.3, 0.2, 0.1]
    
    for i, (start_pos, start_ang) in enumerate(starts):
        if i >= len(time_allocation):
            break
        time_left = DEADLINE - (time.time() - t0)
        if time_left < 5.0:
            break
            
        allocated = total_sa_time * time_allocation[i]
        # Don't allocate more than remaining time minus polish
        allocated = min(allocated, time_left - polish_time - 2.0)
        if allocated <= 0:
            continue
            
        print(f"Running SA start {i} with {allocated:.1f}s")
        res_side, res_pos, res_ang, res_phi, res_cx, res_cy = run_sa(start_pos, start_ang, allocated)
        
        if res_side < best_side:
            best_side = res_side
            best_positions = res_pos
            best_angles = res_ang
            best_phi = res_phi
            best_cx = res_cx
            best_cy = res_cy

    # Polish Phase
    remaining_time = DEADLINE - (time.time() - t0)
    
    if remaining_time > 2.0:
        # 1. Coordinate Descent Polish
        def polish_coordinate_descent(pos, ang, time_limit):
            current_side_cd, _, _, _ = compute_min_outer_side(pos, ang)
            best_side_cd = current_side_cd
            best_pos_cd = pos.copy()
            best_ang_cd = ang.copy()
            
            t_start = time.time()
            step = 0.05
            min_step = 0.0001
            no_improve_cd = 0
            max_no_improve = 30
            
            while time.time() - t_start < time_limit and step > min_step:
                improved = False
                for idx in range(N):
                    if time.time() - t_start > time_limit:
                        return best_side_cd, best_pos_cd, best_ang_cd
                    
                    # Try moving in x, y, and angle dimensions
                    for dim in range(3):
                        for direction in [1, -1]:
                            best_local = best_side_cd
                            best_local_pos = best_pos_cd.copy()
                            best_local_ang = best_ang_cd.copy()
                            
                            # Try multiple step fractions
                            for frac in [1.0, 0.7, 0.4, 0.2]:
                                trial_step = step * frac
                                new_pos = best_pos_cd.copy()
                                new_ang = best_ang_cd.copy()
                                
                                if dim == 0:
                                    new_pos[idx, 0] += direction * trial_step
                                elif dim == 1:
                                    new_pos[idx, 1] += direction * trial_step
                                else:
                                    new_ang[idx] += direction * trial_step
                                
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
                                
                                if new_side < best_local:
                                    best_local = new_side
                                    best_local_pos = new_pos
                                    best_local_ang = new_ang
                            
                            if best_local < best_side_cd - 1e-12:
                                best_pos_cd = best_local_pos
                                best_ang_cd = best_local_ang
                                best_side_cd = best_local
                                improved = True
                                no_improve_cd = 0
                                break
                        
                        if improved:
                            break
                        if time.time() - t_start > time_limit:
                            return best_side_cd, best_pos_cd, best_ang_cd
                    else:
                        continue
                    break
                
                if not improved:
                    no_improve_cd += 1
                    step *= 0.7
                    if no_improve_cd > max_no_improve:
                        break
                else:
                    step *= 0.9
                    no_improve_cd = 0
            
            return best_side_cd, best_pos_cd, best_ang_cd

        cd_time = min(remaining_time * 0.8, 20.0)
        cd_side, cd_pos, cd_ang = polish_coordinate_descent(best_positions, best_angles, cd_time)
        
        if cd_side < best_side:
            best_side = cd_side
            best_positions = cd_pos
            best_angles = cd_ang
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
        
        # 2. Final small SA polish if time remains
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 5.0:
             # Run a very short, high-temperature SA to escape local minima
             res_side, res_pos, res_ang, res_phi, res_cx, res_cy = run_sa(best_positions, best_angles, remaining_time - 2.0)
             if res_side < best_side:
                 best_side = res_side
                 best_positions = res_pos
                 best_angles = res_ang
                 best_phi = res_phi
                 best_cx = res_cx
                 best_cy = res_cy

    # Final verification
    if not is_valid_configuration(best_positions, best_angles):
        best_positions = base_positions.copy()
        best_angles = base_angles.copy()
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, base_angles)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
