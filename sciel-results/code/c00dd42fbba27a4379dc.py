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
        # Max distance between centers for touching hexagons is 2*sqrt(3)
        if dist_sq > 12.0: # (2*sqrt(3))^2 = 12
            return False
        
        a1 = h1[2]
        a2 = h2[2]
        c1, s1 = math.cos(a1), math.sin(a1)
        c2, s2 = math.cos(a2), math.sin(a2)
        
        # Vertices of h1
        v1x = h1[0] + _ux * c1 - _uy * s1
        v1y = h1[1] + _ux * s1 + _uy * c1
        # Vertices of h2
        v2x = h2[0] + _ux * c2 - _uy * s2
        v2y = h2[1] + _ux * s2 + _uy * c2
        
        # Normals of h1
        en1x = _enx * c1 - _eny * s1
        en1y = _enx * s1 + _eny * c1
        # Normals of h2
        en2x = _enx * c2 - _eny * s2
        en2y = _enx * s2 + _eny * c2
        
        # Separation axis theorem
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
        """Fast side-length eval for a single phi angle (radians)."""
        na = phi_rad + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(na)
        ny = np.sin(na)
        proj = offsets[:, 0] * nx[None, :] + offsets[:, 1] * ny[None, :]
        max_projs = np.max(np.abs(proj), axis=0)
        return (2.0 / sqrt3) * np.max(max_projs)

    def compute_outer_side_and_center(positions, angles_deg, refine=True):
        """
        Computes the minimum side length of the outer hexagon containing all inner hexagons.
        Returns (side_length, center_x, center_y, optimal_rotation_deg)
        """
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        
        # Compute centroid
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        # Fully vectorized: compute side length for finer phi angles (0.5 deg resolution)
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
        """Fast side-length evaluation without refine (for SA loop)."""
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

    def smooth_objective(x, temperature=0.05):
        """
        Smooth surrogate objective using log-sum-exp for the max projection.
        x: array of shape (N*3,) with [x0, y0, a0, x1, y1, a1, ...]
        Returns a smooth approximation of the side length + collision penalty.
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
        
        # Smooth max over phi angles using log-sum-exp
        # We'll use a fine grid and soft-max
        phi_vals = np.arange(0, 60, 1.0)
        phi_rad = np.radians(phi_vals)
        normal_angles = phi_rad[:, None] + np.pi/6 + (np.arange(6) * np.pi/3)[None, :]
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        
        proj = offsets[:, 0:1, None] * nx[None, :, :] + offsets[:, 1:2, None] * ny[None, :, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)  # shape (60, 6)
        side_vals = (2.0 / sqrt3) * max_projs  # shape (60, 6)
        
        # For each phi, the side length is max over 6 directions
        # We want min over phi of (max over 6 dirs)
        # Smooth this with log-sum-exp
        
        # First, soft-max over the 6 directions for each phi
        # Then, soft-min over the 60 phi values
        
        # Soft max over 6 directions
        # For each phi, compute soft max of side_vals[phi, :]
        soft_max_dirs = np.zeros(60)
        for i in range(60):
            vals = side_vals[i, :]
            m = np.max(vals)
            soft_max_dirs[i] = m + temperature * np.log(np.sum(np.exp((vals - m) / temperature)))
        
        # Soft min over phi values
        # soft_min = -soft_max(-vals)
        neg_vals = -soft_max_dirs
        m = np.max(neg_vals)
        soft_min_phi = -(m + temperature * np.log(np.sum(np.exp((neg_vals - m) / temperature))))
        
        smooth_side = -soft_min_phi
        
        # Collision penalty (smooth)
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                h1 = (pos[i, 0], pos[i, 1], angles_rad[i])
                h2 = (pos[j, 0], pos[j, 1], angles_rad[j])
                dx = h1[0] - h2[0]
                dy = h1[1] - h2[1]
                dist_sq = dx*dx + dy*dy
                if dist_sq > 12.0:
                    continue
                
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
                
                gaps = []
                for i_ax in range(6):
                    nx, ny = en1x[i_ax], en1y[i_ax]
                    p1 = v1x * nx + v1y * ny
                    p2 = v2x * nx + v2y * ny
                    gap = p2.min() - p1.max()
                    gaps.append(gap)
                for i_ax in range(6):
                    nx, ny = en2x[i_ax], en2y[i_ax]
                    p1 = v1x * nx + v1y * ny
                    p2 = v2x * nx + v2y * ny
                    gap = p2.min() - p1.max()
                    gaps.append(gap)
                
                gaps_arr = np.array(gaps)
                min_gap = np.min(gaps_arr)
                if min_gap < 0:
                    # Overlapping, add penalty
                    penalty += 10.0 * (-min_gap) ** 2
        
        return smooth_side + penalty

    def polish_smooth_lbfgs(pos, ang, time_limit):
        """
        Use L-BFGS-B to polish the configuration using the smooth surrogate.
        """
        x0 = np.zeros(N*3)
        for i in range(N):
            x0[i*3] = pos[i, 0]
            x0[i*3+1] = pos[i, 1]
            x0[i*3+2] = ang[i]
        
        best_x = x0.copy()
        best_val = smooth_objective(x0, 0.05)
        
        t_start = time.time()
        
        # Try multiple temperature values
        for temp in [0.1, 0.05, 0.02, 0.01]:
            if time.time() - t_start > time_limit:
                break
            
            def obj(x):
                return smooth_objective(x, temp)
            
            try:
                res = minimize(
                    obj,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if res.fun < best_val:
                    # Check if the result is actually valid (no collisions)
                    res_pos = np.zeros((N, 2))
                    res_ang = np.zeros(N)
                    for i in range(N):
                        res_pos[i, 0] = res.x[i*3]
                        res_pos[i, 1] = res.x[i*3+1]
                        res_ang[i] = res.x[i*3+2]
                    
                    # Verify no collisions
                    angles_rad = np.radians(res_ang)
                    valid = True
                    for i in range(N):
                        for j in range(i+1, N):
                            h1 = (res_pos[i, 0], res_pos[i, 1], angles_rad[i])
                            h2 = (res_pos[j, 0], res_pos[j, 1], angles_rad[j])
                            if check_collision(h1, h2):
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        best_x = res.x.copy()
                        best_val = res.fun
            
            except Exception:
                pass
        
        # Extract best result
        best_pos = np.zeros((N, 2))
        best_ang = np.zeros(N)
        for i in range(N):
            best_pos[i, 0] = best_x[i*3]
            best_pos[i, 1] = best_x[i*3+1]
            best_ang[i] = best_x[i*3+2]
        
        return best_pos, best_ang

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

    # --- Phase 1.5: Smooth Surrogate Optimization ---
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 15.0:
        try:
            smooth_pos, smooth_ang = polish_smooth_lbfgs(best_positions, best_angles, remaining_time * 0.7)
            
            # Verify and evaluate
            if is_valid(smooth_pos, smooth_ang):
                smooth_side, smooth_cx, smooth_cy, smooth_phi = compute_outer_side_and_center(smooth_pos, smooth_ang)
                if smooth_side < best_side:
                    best_side = smooth_side
                    best_positions = smooth_pos
                    best_angles = smooth_ang
                    best_cx = smooth_cx
                    best_cy = smooth_cy
                    best_phi = smooth_phi
        except Exception:
            pass

    # --- Phase 2: Nelder-Mead Local Refinement ---
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 5.0:
        def nm_objective(x):
            pos = np.zeros((N, 2))
            ang = np.zeros(N)
            for i in range(N):
                pos[i, 0] = x[i*3]
                pos[i, 1] = x[i*3+1]
                ang[i] = x[i*3+2]
            
            angles_rad = np.radians(ang)
            for i in range(N):
                for j in range(i+1, N):
                    h1 = (pos[i, 0], pos[i, 1], angles_rad[i])
                    h2 = (pos[j, 0], pos[j, 1], angles_rad[j])
                    if check_collision(h1, h2):
                        return 100.0
            
            side, _, _, _ = fast_side_length(pos, ang)
            return side
        
        x0 = np.zeros(N*3)
        for i in range(N):
            x0[i*3] = best_positions[i, 0]
            x0[i*3+1] = best_positions[i, 1]
            x0[i*3+2] = best_angles[i]
        
        try:
            nm_result = minimize(
                nm_objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            if nm_result.fun < 100.0:
                nm_pos = np.zeros((N, 2))
                nm_ang = np.zeros(N)
                for i in range(N):
                    nm_pos[i, 0] = nm_result.x[i*3]
                    nm_pos[i, 1] = nm_result.x[i*3+1]
                    nm_ang[i] = nm_result.x[i*3+2]
                
                if nm_result.fun < best_side:
                    best_side = nm_result.fun
                    best_positions = nm_pos
                    best_angles = nm_ang
                    best_side, best_cx, best_cy, best_phi = compute_outer_side_and_center(best_positions, best_angles)
        except Exception:
            pass
        
        # Phase 3: Coordinate Descent final polish
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
                                
                            if time.time() - t_start > time_limit:
                                return best_side_cd, best_pos_cd, best_ang_cd
                            else:
                                continue
                            break
                        else:
                            continue
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
        # Fallback to base if something went wrong
        best_positions = base_positions.copy()
        best_angles = base_angles.copy()
        best_side, best_cx, best_cy, best_phi = compute_outer_side_and_center(best_positions, best_angles)
    
    # Final high-precision polish with tighter tolerances
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 3.0:
        # Multi-stage coordinate descent with adaptive steps
        stages = [(0.01, 30), (0.003, 40), (0.001, 50), (0.0003, 60), (0.0001, 80)]
        for step, max_iters in stages:
            if time.time() - t0 > DEADLINE - 2.0:
                break
            for _ in range(max_iters):
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
                    break
    
    # Recompute final center and phi
    best_side, best_cx, best_cy, best_phi = compute_outer_side_and_center(best_positions, best_angles)
    
    # Final verification and cleanup
    if not is_valid(best_positions, best_angles):
        # Fallback to base if something went wrong
        best_positions = base_positions.copy()
        best_angles = base_angles.copy()
        best_side, best_cx, best_cy, best_phi = compute_outer_side_and_center(best_positions, best_angles)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
