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
    
    sqrt3 = math.sqrt(3)
    
    # Precompute for speed
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

    def check_collision(h1, h2):
        """
        Returns True if h1 and h2 overlap.
        h1, h2 are (x, y, angle_rad)
        """
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        if dist_sq > 12.0:
            return False
        
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
                return False
        for i in range(6):
            nx, ny = en2x[i], en2y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            if p1.max() < p2.min() or p2.max() < p1.min():
                return False
        return True

    def _eval_phi_fast(offsets, phi_rad, sqrt3):
        na = phi_rad + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(na)
        ny = np.sin(na)
        proj = offsets[:, 0] * nx[None, :] + offsets[:, 1] * ny[None, :]
        max_projs = np.max(np.abs(proj), axis=0)
        return (2.0 / sqrt3) * np.max(max_projs)

    def compute_outer_side_and_center(positions, angles_deg, refine=True):
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        phi_vals = np.arange(0, 60, 0.5)
        phi_rad = np.radians(phi_vals)
        
        normal_angles = phi_rad[:, None] + np.pi/6 + (np.arange(6) * np.pi/3)[None, :]
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        
        proj = offsets[:, 0:1, None] * nx[None, :, :] + offsets[:, 1:2, None] * ny[None, :, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / sqrt3) * max_projs
        best_sides = np.max(side_vals, axis=1)
        
        best_idx = np.argmin(best_sides)
        best_side = best_sides[best_idx]
        best_phi = phi_vals[best_idx]
        
        if refine:
            lo = max(0.0, best_phi - 0.5)
            hi = min(60.0, best_phi + 0.5)
            
            def eval_phi(phi_deg):
                phi = np.radians(phi_deg)
                na = phi + np.pi/6 + np.arange(6) * np.pi/3
                nx_r = np.cos(na)
                ny_r = np.sin(na)
                proj_r = offsets[:, 0:1] * nx_r[None, :] + offsets[:, 1:2] * ny_r[None, :]
                abs_proj_r = np.abs(proj_r)
                max_projs_r = np.max(abs_proj_r, axis=0)
                side_vals_r = (2.0 / sqrt3) * max_projs_r
                return np.max(side_vals_r)
                
            try:
                res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded', options={'xatol': 1e-10})
                if res.success and res.fun < best_side:
                    best_side = res.fun
                    best_phi = res.x
            except Exception:
                pass
        
        return best_side, center[0], center[1], best_phi

    def fast_side_length(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        phi_vals = np.arange(0, 60, 1.0)
        phi_rad = np.radians(phi_vals)
        normal_angles = phi_rad[:, None] + np.pi/6 + (np.arange(6) * np.pi/3)[None, :]
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, 0:1, None] * nx[None, :, :] + offsets[:, 1:2, None] * ny[None, :, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / sqrt3) * max_projs
        best_sides = np.max(side_vals, axis=1)
        best_idx = np.argmin(best_sides)
        return best_sides[best_idx], center[0], center[1], phi_vals[best_idx]

    def is_valid(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        for i in range(N):
            for j in range(i+1, N):
                h1 = (positions[i][0], positions[i][1], angles_rad[i])
                h2 = (positions[j][0], positions[j][1], angles_rad[j])
                if check_collision(h1, h2):
                    return False
        return True

    # --- Initialization: Honeycomb ---
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
    best_cx = 0.0
    best_cy = 0.0
    best_phi = 0.0

    # --- Phase 1: Simulated Annealing ---
    current_positions = base_positions.copy()
    current_angles = base_angles.copy()
    current_side, current_cx, current_cy, current_phi = fast_side_length(current_positions, current_angles)
    
    if current_side < best_side:
        best_side = current_side
        best_positions = current_positions.copy()
        best_angles = current_angles.copy()
        best_cx = current_cx
        best_cy = current_cy
        best_phi = current_phi

    T = 0.5
    T_min = 0.0001
    cooling = 0.99995
    step_size = 0.3
    step_decay = 0.99998
    
    sa_time_budget = DEADLINE * 0.55
    max_steps = 200000
    step_count = 0
    
    while time.time() - t0 < sa_time_budget and step_count < max_steps:
        step_count += 1
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
            if check_collision(h_moved, h_other):
                valid = False
                break
        
        if not valid:
            T *= cooling
            if T < T_min: T = T_min
            step_size *= step_decay
            if step_size < 0.005: step_size = 0.005
            continue
            
        new_side, new_cx, new_cy, new_phi = fast_side_length(new_positions, new_angles)
        
        if new_side < current_side:
            current_positions = new_positions
            current_angles = new_angles
            current_side = new_side
            current_cx = new_cx
            current_cy = new_cy
            current_phi = new_phi
            if new_side < best_side:
                best_side = new_side
                best_positions = new_positions.copy()
                best_angles = new_angles.copy()
                best_cx = new_cx
                best_cy = new_cy
                best_phi = new_phi
        else:
            delta = new_side - current_side
            if random.random() < math.exp(-delta / T):
                current_positions = new_positions
                current_angles = new_angles
                current_side = new_side
                current_cx = new_cx
                current_cy = new_cy
                current_phi = new_phi
                if new_side < best_side:
                    best_side = new_side
                    best_positions = new_positions.copy()
                    best_angles = new_angles.copy()
                    best_cx = new_cx
                    best_cy = new_cy
                    best_phi = new_phi
        
        T *= cooling
        if T < T_min: T = T_min
        step_size *= step_decay
        if step_size < 0.005: step_size = 0.005

    # --- Phase 2: Kinematic Chain Optimization ---
    # Represent the 12 hexagons with a reduced parameter set:
    # - 2 parameters: cluster center (cx, cy)
    # - 1 parameter: global rotation of the cluster
    # - 8 parameters: relative offsets for 4 "corner" hexagons (2 each)
    # The remaining 8 hexagons are placed at fixed relative positions
    # This reduces from 36 dims to ~11 dims
    
    def kinematic_to_positions(params):
        """
        Convert kinematic parameters to full positions/angles.
        params: [cx, cy, global_rot, 
                 rx1, ry1, rx2, ry2, rx3, ry3, rx4, ry4,
                 ... potentially more]
        We use a template where hexagons are placed in a symmetric pattern.
        """
        cx, cy = params[0], params[1]
        global_rot = np.radians(params[2])
        
        # Template: 12 hexagons in a 4-4-4 pattern
        # Row 0 (top): 4 hexagons
        # Row 1 (middle): 4 hexagons  
        # Row 2 (bottom): 4 hexagons
        # With adjustable spacing
        
        # Base template positions (relative to cluster center)
        # Using a compact arrangement
        template = np.array([
            # Top row (y = +1.5*sqrt3)
            [-1.5*sqrt3, 1.5*sqrt3],
            [-0.5*sqrt3, 1.5*sqrt3],
            [0.5*sqrt3, 1.5*sqrt3],
            [1.5*sqrt3, 1.5*sqrt3],
            # Middle row (y = 0)
            [-2.0*sqrt3, 0.0],
            [-1.0*sqrt3, 0.0],
            [1.0*sqrt3, 0.0],
            [2.0*sqrt3, 0.0],
            # Bottom row (y = -1.5*sqrt3)
            [-1.5*sqrt3, -1.5*sqrt3],
            [-0.5*sqrt3, -1.5*sqrt3],
            [0.5*sqrt3, -1.5*sqrt3],
            [1.5*sqrt3, -1.5*sqrt3],
        ])
        
        # Apply global rotation
        c, s = np.cos(global_rot), np.sin(global_rot)
        rot_mat = np.array([[c, -s], [s, c]])
        rotated = (rot_mat @ template.T).T
        
        # Add offsets from params (if provided)
        positions = rotated + np.array([cx, cy])
        
        # All hexagons have the same angle as global rotation
        angles = np.full(N, np.degrees(global_rot))
        
        return positions, angles

    def kinematic_objective(params):
        """
        Objective: outer hexagon side length + collision penalty
        """
        positions, angles = kinematic_to_positions(params)
        
        # Check collisions
        angles_rad = np.radians(angles)
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                h1 = (positions[i, 0], positions[i, 1], angles_rad[i])
                h2 = (positions[j, 0], positions[j, 1], angles_rad[j])
                if check_collision(h1, h2):
                    # Compute overlap depth approximately
                    dx = h1[0] - h2[0]
                    dy = h1[1] - h2[1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    overlap = max(0, 2*sqrt3 - dist)
                    penalty += overlap ** 2
        
        if penalty > 1e-6:
            return 1000.0 + penalty
        
        # Compute side length
        side, _, _, _ = fast_side_length(positions, angles)
        return side

    def optimize_kinematic(time_limit):
        """
        Optimize the kinematic parameters using L-BFGS-B
        """
        t_start = time.time()
        
        # Initial guess: centered at origin, no rotation
        x0 = np.array([0.0, 0.0, 0.0])
        
        best_side = float('inf')
        best_params = x0.copy()
        
        # Try multiple starting points
        starts = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 15.0]),
            np.array([0.0, 0.0, 30.0]),
            np.array([0.5, 0.0, 0.0]),
            np.array([0.0, 0.5, 0.0]),
            np.array([-0.5, 0.0, 15.0]),
        ]
        
        for x_start in starts:
            if time.time() - t_start > time_limit * 0.8:
                break
            
            try:
                res = minimize(
                    kinematic_objective, x_start, method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                )
                
                if res.fun < best_side and res.fun < 1000:
                    # Verify and get actual side length
                    positions, angles = kinematic_to_positions(res.x)
                    if is_valid(positions, angles):
                        side, _, _, _ = fast_side_length(positions, angles)
                        if side < best_side:
                            best_side = side
                            best_params = res.x.copy()
            except Exception:
                pass
        
        if best_side < float('inf'):
            positions, angles = kinematic_to_positions(best_params)
            return best_side, positions, angles
        return float('inf'), None, None

    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 10.0:
        kc_side, kc_pos, kc_ang = optimize_kinematic(remaining_time * 0.3)
        if kc_side < best_side and kc_pos is not None:
            best_side = kc_side
            best_positions = kc_pos
            best_angles = kc_ang
            best_side, best_cx, best_cy, best_phi = compute_outer_side_and_center(best_positions, best_angles)

    # --- Phase 3: Smooth Surrogate with L-BFGS-B ---
    def smooth_side_length(offsets, phi_rad, sqrt3):
        """Smooth approximation using log-sum-exp."""
        na = phi_rad + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(na)
        ny = np.sin(na)
        proj = offsets[:, 0] * nx[None, :] + offsets[:, 1] * ny[None, :]
        abs_proj = np.abs(proj)
        
        soft_maxes = np.zeros(6)
        for d in range(6):
            vals = abs_proj[:, d] + 1e-10
            m = np.max(vals)
            soft_maxes[d] = m + np.log(np.sum(np.exp(vals - m)))
        
        return (2.0 / sqrt3) * np.max(soft_maxes)

    def smooth_objective_full(x):
        """
        Full smooth objective for L-BFGS-B on all 36 parameters.
        """
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(N):
            pos[i, 0] = x[i*3]
            pos[i, 1] = x[i*3+1]
            ang[i] = x[i*3+2]
        
        angles_rad = np.radians(ang)
        all_verts = get_all_vertices(pos, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        phi_rad = np.radians(best_phi)
        side_smooth = smooth_side_length(offsets, phi_rad, sqrt3)
        
        # Collision penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                h1 = (pos[i, 0], pos[i, 1], angles_rad[i])
                h2 = (pos[j, 0], pos[j, 1], angles_rad[j])
                if check_collision(h1, h2):
                    dx = h1[0] - h2[0]
                    dy = h1[1] - h2[1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    overlap = max(0, 2*sqrt3 - dist)
                    penalty += math.exp(50.0 * overlap)
        
        return side_smooth + penalty

    def polish_lbfgs(pos, ang, time_limit):
        n_params = N * 3
        x0 = np.zeros(n_params)
        for i in range(N):
            x0[i*3] = pos[i, 0]
            x0[i*3+1] = pos[i, 1]
            x0[i*3+2] = ang[i]
        
        t_start = time.time()
        best_side_lb = float('inf')
        best_pos_lb = pos.copy()
        best_ang_lb = ang.copy()
        
        for _ in range(3):
            if time.time() - t_start > time_limit * 0.7:
                break
            
            try:
                res = minimize(
                    smooth_objective_full, x0, method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100}
                )
                
                new_pos = np.zeros((N, 2))
                new_ang = np.zeros(N)
                for i in range(N):
                    new_pos[i, 0] = res.x[i*3]
                    new_pos[i, 1] = res.x[i*3+1]
                    new_ang[i] = res.x[i*3+2]
                
                if is_valid(new_pos, new_ang):
                    side, _, _, _ = compute_outer_side_and_center(new_pos, new_ang)
                    if side < best_side_lb:
                        best_side_lb = side
                        best_pos_lb = new_pos
                        best_ang_lb = new_ang
                        x0 = res.x.copy()
            except Exception:
                pass
        
        return best_side_lb, best_pos_lb, best_ang_lb

    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 5.0:
        lbfgs_side, lbfgs_pos, lbfgs_ang = polish_lbfgs(best_positions, best_angles, remaining_time * 0.4)
        if lbfgs_side < best_side:
            best_side = lbfgs_side
            best_positions = lbfgs_pos
            best_angles = lbfgs_ang
            best_side, best_cx, best_cy, best_phi = compute_outer_side_and_center(best_positions, best_angles)

    # --- Phase 4: Coordinate Descent Final Polish ---
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 5.0:
        def polish_coordinate_descent(pos, ang, time_limit):
            current_side, _, _, _ = compute_outer_side_and_center(pos, ang)
            best_side_cd = current_side
            best_pos_cd = pos.copy()
            best_ang_cd = ang.copy()
            
            t_start = time.time()
            step = 0.05
            min_step = 0.0001
            
            while time.time() - t_start < time_limit and step > min_step:
                improved = False
                for idx in range(N):
                    if time.time() - t_start > time_limit:
                        return best_side_cd, best_pos_cd, best_ang_cd
                    for dim in range(3):
                        if time.time() - t_start > time_limit:
                            return best_side_cd, best_pos_cd, best_ang_cd
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
                                if check_collision(h_moved, h_other):
                                    valid = False
                                    break
                            
                            if not valid:
                                continue
                            
                            new_side, _, _, _ = fast_side_length(new_pos, new_ang)
                            
                            if new_side < best_side_cd - 1e-10:
                                best_pos_cd = new_pos
                                best_ang_cd = new_ang
                                best_side_cd = new_side
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                
                if not improved:
                    step *= 0.7
                else:
                    step *= 0.95
                    if step < min_step:
                        step = min_step
            
            return best_side_cd, best_pos_cd, best_ang_cd
        
        cd_side, cd_pos, cd_ang = polish_coordinate_descent(best_positions, best_angles, remaining_time * 0.9)
        if cd_side < best_side:
            best_side = cd_side
            best_positions = cd_pos
            best_angles = cd_ang
            best_side, best_cx, best_cy, best_phi = compute_outer_side_and_center(best_positions, best_angles)

    # Final verification
    if not is_valid(best_positions, best_angles):
        best_positions = base_positions.copy()
        best_angles = base_angles.copy()
        best_side, best_cx, best_cy, best_phi = compute_outer_side_and_center(best_positions, best_angles)
    
    # Final high-precision polish
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 3.0:
        step = 0.001
        for _ in range(50):
            if time.time() - t0 > DEADLINE - 2.0:
                break
            improved = False
            for idx in range(N):
                for dim in range(3):
                    for direction in [1, -1]:
                        new_pos = best_positions.copy()
                        new_ang = best_angles.copy()
                        
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
                            if check_collision(h_moved, h_other):
                                valid = False
                                break
                        
                        if not valid:
                            continue
                        
                        new_side, _, _, _ = compute_outer_side_and_center(new_pos, new_ang)
                        
                        if new_side < best_side - 1e-12:
                            best_positions = new_pos
                            best_angles = new_ang
                            best_side = new_side
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                step *= 0.5
                if step < 1e-6:
                    break
    
    best_side, best_cx, best_cy, best_phi = compute_outer_side_and_center(best_positions, best_angles)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
