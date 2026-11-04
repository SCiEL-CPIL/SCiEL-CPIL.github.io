import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0  # Safety margin before 150s wall-clock kill
sqrt3 = math.sqrt(3)

# Precompute unit hexagon vertices and edge normals (angle 0)
# Vertices at angles 0, 60, 120, 180, 240, 300
_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))

# Edge normals are perpendicular to edges.
# Edge 0 connects vertex 0 and 1.
# Normal should point outwards.
_EDGE_NORMALS = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i+1)%6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i+1)%6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, _ex
    _norm = math.sqrt(_nx*_nx + _ny*_ny)
    # Ensure normal points outwards (dot product with vertex should be positive)
    _dot = _nx * _UNIT_HEX_VERTS[_i, 0] + _ny * _UNIT_HEX_VERTS[_i, 1]
    if _dot < 0:
        _nx, _ny = -_nx, -_ny
    _EDGE_NORMALS[_i] = [_nx / _norm, _ny / _norm]

def construct():
    t0 = time.time()
    
    unit_hex_verts = _UNIT_HEX_VERTS
    edge_normals = _EDGE_NORMALS
    
    # Precompute flat arrays for speed
    _ux = unit_hex_verts[:, 0]
    _uy = unit_hex_verts[:, 1]
    _enx = edge_normals[:, 0]
    _eny = edge_normals[:, 1]
    
    def get_all_vertices(positions, angles_rad):
        """
        positions: (N, 2)
        angles_rad: (N,)
        returns: (N, 6, 2)
        """
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        # Rotate vertices
        rx = _ux[None, :] * c[:, None] - _uy[None, :] * s[:, None]
        ry = _ux[None, :] * s[:, None] + _uy[None, :] * c[:, None]
        # Translate
        verts_x = rx + positions[:, 0:1]
        verts_y = ry + positions[:, 1:2]
        return np.stack((verts_x, verts_y), axis=-1)

    def hexagons_disjoint_fast(h1, h2):
        """
        h1, h2: (x, y, angle_rad)
        Returns True if disjoint.
        """
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        # Max distance between centers of two unit hexagons touching is 2*sqrt(3) ~ 3.46
        # Square is 12. If dist_sq > 12, they are definitely disjoint.
        if dist_sq > 12.0:
            return True
        # Bounding box check
        # Max extent of a unit hexagon is 1 (x) and sqrt(3)/2 (y) if flat top?
        # Actually, side 1 hexagon: width 2, height sqrt(3).
        # If rotated, max extent is 1 in any direction from center? No, vertex distance is 1.
        # So if |dx| > 2 or |dy| > 2, they are disjoint.
        if abs(dx) > 2.0 or abs(dy) > 2.0:
            return True
            
        a1 = h1[2]
        a2 = h2[2]
        c1, s1 = math.cos(a1), math.sin(a1)
        c2, s2 = math.cos(a2), math.sin(a2)
        
        # Compute vertices
        v1x = h1[0] + _ux * c1 - _uy * s1
        v1y = h1[1] + _ux * s1 + _uy * c1
        v2x = h2[0] + _ux * c2 - _uy * s2
        v2y = h2[1] + _ux * s2 + _uy * c2
        
        # Compute normals
        en1x = _enx * c1 - _eny * s1
        en1y = _enx * s1 + _eny * c1
        en2x = _enx * c2 - _eny * s2
        en2y = _enx * s2 + _eny * c2
        
        # Check axes from h1
        for i in range(6):
            nx, ny = en1x[i], en1y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            if p1.max() < p2.min() or p2.max() < p1.min():
                return True
        
        # Check axes from h2
        for i in range(6):
            nx, ny = en2x[i], en2y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            if p1.max() < p2.min() or p2.max() < p1.min():
                return True
        
        return False

    def compute_min_outer_side_fast(positions, angles_deg):
        """
        Fast vectorized evaluation of the minimum outer hexagon side length.
        Returns: (side_length, best_phi_deg, center_x, center_y)
        """
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        # Sample phi from 0 to 60 degrees with 1 degree resolution
        phi_vals = np.radians(np.arange(0, 60, 1.0))
        # Normal angles for each phi and each of the 6 edges
        # Outer hexagon edge normals are at phi + 30 + k*60
        normal_angles = phi_vals[:, None] + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        
        # proj[v, p, e] = offset_v dot normal_{p,e}
        # offsets: (72, 2)
        # nx, ny: (60, 6)
        proj = offsets[:, None, None, 0] * nx[None, :, :, None] + offsets[:, None, None, 1] * ny[None, :, :, None]
        abs_proj = np.abs(proj)
        
        # Max projection over vertices for each (phi, edge)
        max_projs = np.max(abs_proj, axis=0)  # (60, 6)
        
        # Side length is (2/sqrt(3)) * max_projection
        # We want the minimum side length over phi, which is (2/sqrt(3)) * min_phi (max_e max_v proj)
        side_vals = (2.0 / sqrt3) * np.max(max_projs, axis=1)  # (60,)
        
        best_idx = np.argmin(side_vals)
        best_side = float(side_vals[best_idx])
        best_phi = float(best_idx)
        
        return best_side, best_phi, center[0], center[1]

    def smooth_outer_side_surrogate(positions, angles_deg, beta=50.0):
        """
        Smooth log-sum-exp surrogate for the outer hexagon side length.
        """
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        # Sample phi values densely (0.5 deg resolution)
        phi_vals = np.radians(np.arange(0, 60, 0.5))
        normal_angles = phi_vals[:, None] + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        
        proj = offsets[:, None, None, 0] * nx[None, :, :, None] + offsets[:, None, None, 1] * ny[None, :, :, None]
        abs_proj = np.abs(proj)  # (72, 120, 6)
        
        # Soft-max over vertices (axis 0)
        max_v = abs_proj.max(axis=0, keepdims=True)
        logsumexp_v = max_v.squeeze(0) + np.log(np.sum(np.exp(beta * (abs_proj - max_v)), axis=0)) / beta
        
        # Soft-max over edge directions (axis 1)
        max_e = logsumexp_v.max(axis=1, keepdims=True)
        logsumexp_e = max_e.squeeze(1) + np.log(np.sum(np.exp(beta * (logsumexp_v - max_e)), axis=1)) / beta
        
        # Soft-max over phi (axis 0)
        max_p = logsumexp_e.max(axis=0, keepdims=True)
        logsumexp_phi = max_p.squeeze(0) + np.log(np.sum(np.exp(beta * (logsumexp_e - max_p)), axis=0)) / beta
        
        return (2.0 / sqrt3) * logsumexp_phi

    def is_valid_configuration(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        for i in range(N):
            for j in range(i+1, N):
                h1 = (positions[i][0], positions[i][1], angles_rad[i])
                h2 = (positions[j][0], positions[j][1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    return False
        return True

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

    best_side = float('inf')
    best_positions = base_positions.copy()
    best_angles = base_angles.copy()
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    # SA Phase
    sa_time_budget = DEADLINE * 0.7
    
    # Multiple initial configurations for restarts
    def make_honeycomb(scale=1.0, offset_angle=0.0):
        a1 = np.array([sqrt3, 0.0]) * scale
        a2 = np.array([sqrt3/2.0, 1.5]) * scale
        positions = np.zeros((N, 2))
        angles = np.zeros(N) + offset_angle
        for idx, (i, j) in enumerate(lattice_coords):
            positions[idx] = i * a1 + j * a2
        return positions, angles
    
    def make_ring_config():
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for i in range(N):
            theta = 2 * np.pi * i / N
            r = 2.5
            positions[i, 0] = r * math.cos(theta)
            positions[i, 1] = r * math.sin(theta)
            angles[i] = theta * 180 / math.pi
        return positions, angles

    initial_configs = [
        make_honeycomb(1.0, 0.0),
        make_honeycomb(1.0, 15.0),
        make_ring_config(),
    ]
    
    num_restarts = 3
    restart_time = sa_time_budget / num_restarts

    for restart in range(num_restarts):
        if time.time() - t0 >= sa_time_budget:
            break
        
        init_pos, init_ang = initial_configs[restart % len(initial_configs)]
        current_positions = init_pos.copy()
        current_angles = init_ang.copy()
        current_side, current_phi, current_cx, current_cy = compute_min_outer_side_fast(current_positions, current_angles)
        
        if current_side < best_side:
            best_side = current_side
            best_positions = current_positions.copy()
            best_angles = current_angles.copy()
            best_phi = current_phi
            best_cx = current_cx
            best_cy = current_cy

        T = 0.3
        T_min = 0.0001
        cooling = 0.99995
        step_size = 0.2
        
        step_count = 0
        accept_count = 0
        total_count = 0
        
        restart_deadline = t0 + (restart + 1) * restart_time
        
        while time.time() < restart_deadline and time.time() - t0 < sa_time_budget:
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
                step_size *= 0.9999
                if step_size < 0.01: step_size = 0.01
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
            if total_count % 200 == 0:
                acc_rate = accept_count / total_count
                if acc_rate < 0.1:
                    step_size *= 0.8
                    if step_size < 0.01: step_size = 0.01
                elif acc_rate > 0.5:
                    step_size *= 1.1
                    if step_size > 0.5: step_size = 0.5
                accept_count = 0
                total_count = 0
                
            T *= cooling
            if T < T_min: T = T_min
            step_size *= 0.9999
            if step_size < 0.01: step_size = 0.01

    # Polish Phase
    remaining_time = DEADLINE - (time.time() - t0)
    
    if remaining_time > 2.0:
        # L-BFGS-B Polish with smooth surrogate
        def polish_lbfgs(pos, ang, time_limit):
            def objective_smooth(v):
                p = v[:2*N].reshape(N, 2)
                a = v[2*N:3*N]
                # Check validity
                angles_rad = np.radians(a)
                for i in range(N):
                    for j in range(i+1, N):
                        h1 = (p[i][0], p[i][1], angles_rad[i])
                        h2 = (p[j][0], p[j][1], angles_rad[j])
                        if not hexagons_disjoint_fast(h1, h2):
                            return 1e6 # Penalty
                return smooth_outer_side_surrogate(p, a, beta=50.0)
            
            v0 = np.concatenate([pos.flatten(), ang])
            bounds = [(-10, 10)] * (2*N) + [(0, 360)] * N
            
            best_result = None
            best_obj = float('inf')
            
            try:
                res = minimize(objective_smooth, v0, method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-8, 'maxfun': 200})
                if res.fun < 1e5:
                    p_new = res.x[:2*N].reshape(N, 2)
                    a_new = res.x[2*N:3*N]
                    # Verify with fast exact method
                    exact_side, _, _, _ = compute_min_outer_side_fast(p_new, a_new)
                    if exact_side < best_obj:
                        best_obj = exact_side
                        best_result = (exact_side, p_new, a_new)
            except Exception:
                pass
            
            if best_result is not None:
                return best_result
            return None, pos, ang

        lbfgs_time = min(remaining_time * 0.6, 15.0)
        lbfgs_side, lbfgs_pos, lbfgs_ang = polish_lbfgs(best_positions, best_angles, lbfgs_time)
        if lbfgs_side is not None and lbfgs_side < best_side:
            best_side = lbfgs_side
            best_positions = lbfgs_pos
            best_angles = lbfgs_ang

        # Final fine-tune of the outer hexagon orientation
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side_fast(best_positions, best_angles)

    # Final verification and safety check
    elapsed = time.time() - t0
    if elapsed > DEADLINE - 2.0:
        pass
    else:
        if not is_valid_configuration(best_positions, best_angles):
            best_positions = base_positions.copy()
            best_angles = base_angles.copy()
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side_fast(best_positions, best_angles)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle

