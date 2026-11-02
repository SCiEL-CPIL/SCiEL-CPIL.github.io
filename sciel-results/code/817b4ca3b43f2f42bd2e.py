import math
import time
import random
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 12
DEADLINE = 145.0  # Safety margin before 150s wall-clock kill

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
    
    def get_all_vertices(positions, angles_rad):
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        
        ux = unit_hex_verts[:, 0]
        uy = unit_hex_verts[:, 1]
        
        rx = ux[None, :] * c[:, None] - uy[None, :] * s[:, None]
        ry = ux[None, :] * s[:, None] + uy[None, :] * c[:, None]
        
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
        
        # Conservative early exit
        if dist_sq > 4.5:
            return True
            
        a1 = h1[2]
        a2 = h2[2]
        
        c1, s1 = math.cos(a1), math.sin(a1)
        c2, s2 = math.cos(a2), math.sin(a2)
        
        ux = unit_hex_verts[:, 0]
        uy = unit_hex_verts[:, 1]
        
        v1x = h1[0] + ux * c1 - uy * s1
        v1y = h1[1] + ux * s1 + uy * c1
        v2x = h2[0] + ux * c2 - uy * s2
        v2y = h2[1] + ux * s2 + uy * c2
        
        en = edge_normals
        en1x = en[:, 0] * c1 - en[:, 1] * s1
        en1y = en[:, 0] * s1 + en[:, 1] * c1
        en2x = en[:, 0] * c2 - en[:, 1] * s2
        en2y = en[:, 0] * s2 + en[:, 1] * c2
        
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
        best_side = float('inf')
        best_phi = 0.0
        
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
        
        # Coarse scan
        for phi_deg in range(0, 60, 2):
            s = eval_phi(phi_deg)
            if s < best_side:
                best_side = s
                best_phi = phi_deg
                
        # Fine scan
        for phi_deg in np.arange(best_phi - 2, best_phi + 2, 0.05):
            if phi_deg < 0 or phi_deg >= 60: continue
            s = eval_phi(phi_deg)
            if s < best_side:
                best_side = s
                best_phi = phi_deg

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

    # --- Phase 1: Symmetric Initialization ---
    # Model: 2 rings of 6 hexagons.
    # Params: r1, theta1, alpha1, r2, theta2, alpha2
    # r1, r2: radii
    # theta1, theta2: angular offset of the ring
    # alpha1, alpha2: rotation of individual hexagons
    
    def symmetric_objective(params):
        r1, theta1, alpha1, r2, theta2, alpha2 = params
        
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        
        # Inner ring (indices 0-5)
        for i in range(6):
            ang = theta1 + i * np.pi / 3
            positions[i, 0] = r1 * math.cos(ang)
            positions[i, 1] = r1 * math.sin(ang)
            angles[i] = np.degrees(alpha1) # Keep angles in degrees for consistency with other parts? 
            # Actually, let's keep everything in radians internally for the symmetric part, then convert.
            # But the rest of the code uses degrees for angles. Let's stick to degrees in the main loop.
            # For the symmetric objective, let's work in radians for geometry, but return side length.
            
        # Outer ring (indices 6-11)
        for i in range(6):
            ang = theta2 + i * np.pi / 3
            positions[i+6, 0] = r2 * math.cos(ang)
            positions[i+6, 1] = r2 * math.sin(ang)
            angles[i+6] = np.degrees(alpha2)
            
        # Check validity
        angles_rad = np.radians(angles)
        for i in range(N):
            for j in range(i+1, N):
                h1 = (positions[i, 0], positions[i, 1], angles_rad[i])
                h2 = (positions[j, 0], positions[j, 1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    return 1e6 # Penalty
            
        side, phi, cx, cy = compute_min_outer_side(positions, angles)
        return side

    # Bounds for symmetric params
    # r1 ~ 1.7 to 2.5, r2 ~ 3.0 to 4.5
    # theta ~ 0 to 60 deg (pi/3)
    # alpha ~ 0 to 60 deg
    bounds_sym = [
        (1.5, 2.5), # r1
        (0, math.pi/3), # theta1
        (0, math.pi/3), # alpha1
        (2.8, 4.0), # r2
        (0, math.pi/3), # theta2
        (0, math.pi/3)  # alpha2
    ]
    
    best_sym_side = float('inf')
    best_sym_params = None
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0
    
    # Run differential evolution for symmetric model
    if time.time() - t0 < DEADLINE * 0.4:
        try:
            res_sym = differential_evolution(
                symmetric_objective, 
                bounds_sym, 
                maxiter=100, 
                popsize=10, 
                tol=0.01,
                seed=42
            )
            if res_sym.fun < 1e5:
                best_sym_params = res_sym.x
                # Reconstruct best positions
                r1, theta1, alpha1, r2, theta2, alpha2 = best_sym_params
                positions = np.zeros((N, 2))
                angles = np.zeros(N)
                for i in range(6):
                    ang = theta1 + i * np.pi / 3
                    positions[i, 0] = r1 * math.cos(ang)
                    positions[i, 1] = r1 * math.sin(ang)
                    angles[i] = np.degrees(alpha1)
                for i in range(6):
                    ang = theta2 + i * np.pi / 3
                    positions[i+6, 0] = r2 * math.cos(ang)
                    positions[i+6, 1] = r2 * math.sin(ang)
                    angles[i+6] = np.degrees(alpha2)
                
                if is_valid_configuration(positions, angles):
                    side, phi, cx, cy = compute_min_outer_side(positions, angles)
                    best_sym_side = side
                    best_positions = positions
                    best_angles = angles
                    best_phi = phi
                    best_cx = cx
                    best_cy = cy
        except Exception:
            pass

    # Fallback initialization if symmetric failed
    if best_positions is None:
        sqrt3 = math.sqrt(3)
        a1 = np.array([sqrt3, 0.0])
        a2 = np.array([sqrt3/2.0, 1.5])
        lattice_coords = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, -1), (-1, 1), (2, -1), (-2, 1),
            (1, 1), (-1, -1), (2, 0),
        ]
        best_positions = np.zeros((N, 2))
        best_angles = np.zeros(N)
        for idx, (i, j) in enumerate(lattice_coords):
            best_positions[idx] = i * a1 + j * a2
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    # --- Phase 2: General Optimization (SA + NM) ---
    
    current_positions = best_positions.copy()
    current_angles = best_angles.copy()
    current_side, current_phi, current_cx, current_cy = compute_min_outer_side(current_positions, current_angles)
    
    best_side = current_side
    best_positions = current_positions.copy()
    best_angles = current_angles.copy()
    best_phi = current_phi
    best_cx = current_cx
    best_cy = current_cy

    T = 0.5
    T_min = 0.0005
    cooling = 0.9997
    step_size = 0.3
    step_decay = 0.9999
    
    step_count = 0
    max_steps = 80000
    
    # SA Loop
    while time.time() - t0 < DEADLINE * 0.8 and step_count < max_steps:
        step_count += 1
        
        idx = random.randint(0, N - 1)
        new_positions = current_positions.copy()
        new_angles = current_angles.copy()
        
        dx = random.gauss(0, step_size)
        dy = random.gauss(0, step_size)
        da = random.gauss(0, 8.0)
        
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

    # Polish Step
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
        
    polish_time_budget = DEADLINE - (time.time() - t0)
    if polish_time_budget > 5.0:
        try:
            res = minimize(polish_objective, init_params, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-7})
            
            if res.fun < best_side:
                new_pos = np.zeros((N, 2))
                new_ang = np.zeros(N)
                for i in range(N):
                    new_pos[i, 0] = res.x[3*i]
                    new_pos[i, 1] = res.x[3*i+1]
                    new_ang[i] = res.x[3*i+2]
                
                if is_valid_configuration(new_pos, new_ang):
                    best_side = res.fun
                    best_positions = new_pos
                    best_angles = new_ang
                    best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
                    
                    if time.time() - t0 < DEADLINE - 2.0:
                        init_params2 = np.zeros(3*N)
                        for i in range(N):
                            init_params2[3*i] = best_positions[i, 0]
                            init_params2[3*i+1] = best_positions[i, 1]
                            init_params2[3*i+2] = best_angles[i]
                        
                        res2 = minimize(polish_objective, init_params2, method='Nelder-Mead', 
                                       options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-9})
                        
                        if res2.fun < best_side:
                            new_pos2 = np.zeros((N, 2))
                            new_ang2 = np.zeros(N)
                            for i in range(N):
                                new_pos2[i, 0] = res2.x[3*i]
                                new_pos2[i, 1] = res2.x[3*i+1]
                                new_ang2[i] = res2.x[3*i+2]
                            
                            if is_valid_configuration(new_pos2, new_ang2):
                                best_side = res2.fun
                                best_positions = new_pos2
                                best_angles = new_ang2
                                best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
        except Exception:
            pass

    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle

# EVOLVE-BLOCK-END
