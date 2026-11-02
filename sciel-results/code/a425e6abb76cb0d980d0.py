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
# Edge normals: for each edge from vert i to vert (i+1)%6
_edge_normals_raw = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i+1)%6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i+1)%6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, _ex
    _norm = math.sqrt(_nx*_nx + _ny*_ny)
    _edge_normals_raw[_i] = [_nx / _norm, _ny / _norm]
_EDGE_NORMALS = _edge_normals_raw  # (6, 2) unit normals for edges of unit hex at angle 0

def construct():
    t0 = time.time()
    
    unit_hex_verts = _UNIT_HEX_VERTS
    edge_normals = _EDGE_NORMALS
    
    # Precompute for fast SAT
    _ux = unit_hex_verts[:, 0]
    _uy = unit_hex_verts[:, 1]
    _enx = edge_normals[:, 0]
    _eny = edge_normals[:, 1]
    
    # Helper: Get vertices of all hexagons given positions and angles
    def get_all_vertices(positions, angles_rad):
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        
        rx = _ux[None, :] * c[:, None] - _uy[None, :] * s[:, None]
        ry = _ux[None, :] * s[:, None] + _uy[None, :] * c[:, None]
        
        verts_x = rx + positions[:, 0:1]
        verts_y = ry + positions[:, 1:2]
        
        all_verts = np.stack((verts_x, verts_y), axis=-1)
        return all_verts

    def hexagons_disjoint_fast(h1, h2):
        """
        Fast SAT check for two unit hexagons.
        h1, h2: (x, y, angle_rad)
        """
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
        """
        Compute the minimum side length of a regular hexagon containing all inner hexagons.
        Returns (side_length, optimal_outer_angle_deg, center_x, center_y)
        """
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        sqrt3 = math.sqrt(3)
        
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
        
        # Batch evaluation for speed
        phi_grid = np.arange(0, 60, 0.5)
        best_side = float('inf')
        best_phi = 0.0
        
        for phi_deg in phi_grid:
            s = eval_phi(phi_deg)
            if s < best_side:
                best_side = s
                best_phi = phi_deg
        
        # Refine around best
        res = minimize_scalar(eval_phi, bounds=(best_phi - 1.0, best_phi + 1.0), method='bounded', options={'xatol': 1e-5})
        if res.success and res.fun < best_side:
            best_side = res.fun
            best_phi = res.x

        return best_side, best_phi, center[0], center[1]

    def is_valid_configuration(positions, angles_deg):
        """Check if all hexagons are pairwise disjoint."""
        angles_rad = np.radians(angles_deg)
        for i in range(N):
            for j in range(i+1, N):
                h1 = (positions[i][0], positions[i][1], angles_rad[i])
                h2 = (positions[j][0], positions[j][1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    return False
        return True

    # Initial Configuration: Honeycomb packing
    sqrt3 = math.sqrt(3)
    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3/2.0, 1.5])
    
    lattice_coords = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1),
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
    
    # === PHASE 1: Rigid-Body Relaxation ===
    # Global translation + rotation of the entire cluster
    def rigid_body_objective(params):
        tx, ty, theta_deg = params
        theta = np.radians(theta_deg)
        ct, st = math.cos(theta), math.sin(theta)
        
        # Rotate and translate all positions
        new_pos = np.zeros((N, 2))
        for i in range(N):
            px, py = base_positions[i]
            new_pos[i, 0] = px * ct - py * st + tx
            new_pos[i, 1] = px * st + py * ct + ty
        
        # All angles rotated by theta
        new_angles = base_angles + theta_deg
        
        # Check validity
        if not is_valid_configuration(new_pos, new_angles):
            return 1e6
        
        side, phi, cx, cy = compute_min_outer_side(new_pos, new_angles)
        return side
    
    # Try several initial rigid-body transforms
    rigid_candidates = [
        [0, 0, 0],
        [0, 0, 5],
        [0, 0, 10],
        [0, 0, 15],
        [0, 0, 30],
        [0, 0, 45],
        [0.1, 0, 0],
        [-0.1, 0, 0],
        [0, 0.1, 0],
        [0, -0.1, 0],
    ]
    
    for cand in rigid_candidates:
        if time.time() - t0 > DEADLINE * 0.3:
            break
        try:
            res = minimize(rigid_body_objective, cand, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
            if res.fun < best_side and res.fun < 1e5:
                tx, ty, theta_deg = res.x
                theta = np.radians(theta_deg)
                ct, st = math.cos(theta), math.sin(theta)
                new_pos = np.zeros((N, 2))
                new_ang = base_angles + theta_deg
                for i in range(N):
                    px, py = base_positions[i]
                    new_pos[i, 0] = px * ct - py * st + tx
                    new_pos[i, 1] = px * st + py * ct + ty
                if is_valid_configuration(new_pos, new_ang):
                    best_side = res.fun
                    best_positions = new_pos
                    best_angles = new_ang
                    _, best_phi, best_cx, best_cy = compute_min_outer_side(new_pos, new_ang)
        except Exception:
            pass
    
    # === PHASE 2: SA with Individual Hexagon Moves ===
    num_restarts = 3
    time_per_restart = (DEADLINE * 0.7) / num_restarts
    
    for r in range(num_restarts):
        if time.time() - t0 > DEADLINE * 0.7:
            break
            
        current_positions = best_positions.copy()
        current_angles = best_angles.copy()
        if r > 0:
            current_positions += np.random.normal(0, 0.15, size=(N, 2))
            current_angles += np.random.normal(0, 5.0, size=N)
            
        current_side, current_phi, current_cx, current_cy = compute_min_outer_side(current_positions, current_angles)
        
        if current_side < best_side:
            best_side = current_side
            best_positions = current_positions.copy()
            best_angles = current_angles.copy()
            best_phi = current_phi
            best_cx = current_cx
            best_cy = current_cy

        T = 0.3
        T_min = 0.0005
        cooling = 0.9998
        step_size = 0.25
        step_decay = 0.99995
        
        step_count = 0
        max_steps = 25000
        
        while time.time() - t0 < DEADLINE * 0.7 and step_count < max_steps:
            step_count += 1
            
            idx = random.randint(0, N - 1)
            new_positions = current_positions.copy()
            new_angles = current_angles.copy()
            
            dx = random.gauss(0, step_size)
            dy = random.gauss(0, step_size)
            da = random.gauss(0, 6.0)
            
            new_positions[idx, 0] += dx
            new_positions[idx, 1] += dy
            new_angles[idx] += da
            
            angles_rad_new = np.radians(new_angles)
            valid = True
            h_moved = (new_positions[idx, 0], new_positions[idx, 1], angles_rad_new[idx])
            for j in range(N):
                if j == idx: continue
                h_other = (new_positions[j, 0], new_positions[j, 1], angles_rad_new[j])
                if not hexagons_disjoint_fast(h_moved, h_other):
                    valid = False
                    break
                    
            if not valid:
                T *= cooling
                if T < T_min: T = T_min
                step_size *= step_decay
                if step_size < 0.01: step_size = 0.01
                continue
                
            new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_positions, new_angles)
            
            if new_side < current_side:
                current_positions = new_positions
                current_angles = new_angles
                current_side = new_side
                current_phi = new_phi
                current_cx = new_cx
                current_cy = new_cy
                
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
                    
                    if new_side < best_side:
                        best_side = new_side
                        best_positions = new_positions.copy()
                        best_angles = new_angles.copy()
                        best_phi = new_phi
                        best_cx = new_cx
                        best_cy = new_cy
                        
            T *= cooling
            if T < T_min: T = T_min
            step_size *= step_decay
            if step_size < 0.01: step_size = 0.01

    # === PHASE 3: Coordinate Descent Polish ===
    def polish_coordinate_descent():
        pos = best_positions.copy()
        ang = best_angles.copy()
        side, phi, cx, cy = compute_min_outer_side(pos, ang)
        
        # Coordinate descent on each hexagon's position and angle
        step_sizes = [0.05, 0.03, 0.02, 0.01, 0.005, 0.002, 0.001]
        
        for ss in step_sizes:
            improved = True
            while improved and time.time() - t0 < DEADLINE - 5.0:
                improved = False
                for i in range(N):
                    # Try small perturbations
                    for dx in [-ss, 0, ss]:
                        for dy in [-ss, 0, ss]:
                            for da in [-ss*20, 0, ss*20]:
                                if dx == 0 and dy == 0 and da == 0:
                                    continue
                                new_pos = pos.copy()
                                new_ang = ang.copy()
                                new_pos[i, 0] += dx
                                new_pos[i, 1] += dy
                                new_ang[i] += da
                                
                                angles_rad = np.radians(new_ang)
                                valid = True
                                h_moved = (new_pos[i, 0], new_pos[i, 1], angles_rad[i])
                                for j in range(N):
                                    if j == i: continue
                                    h_other = (new_pos[j, 0], new_pos[j, 1], angles_rad[j])
                                    if not hexagons_disjoint_fast(h_moved, h_other):
                                        valid = False
                                        break
                                
                                if not valid:
                                    continue
                                
                                new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_pos, new_ang)
                                if new_side < side:
                                    pos = new_pos
                                    ang = new_ang
                                    side = new_side
                                    phi = new_phi
                                    cx = new_cx
                                    cy = new_cy
                                    improved = True
                                    if side < best_side:
                                        best_side = side
                                        best_positions = pos.copy()
                                        best_angles = ang.copy()
                                        best_phi = phi
                                        best_cx = cx
                                        best_cy = cy
                    if time.time() - t0 > DEADLINE - 5.0:
                        break
                if time.time() - t0 > DEADLINE - 5.0:
                    break
        
        return best_side, best_positions, best_angles, best_phi, best_cx, best_cy
    
    polish_time_budget = DEADLINE - (time.time() - t0)
    if polish_time_budget > 5.0:
        try:
            best_side, best_positions, best_angles, best_phi, best_cx, best_cy = polish_coordinate_descent()
        except Exception:
            pass

    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
