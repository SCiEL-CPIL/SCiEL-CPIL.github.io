import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0  # Safety margin before 150s wall-clock kill
SQRT3 = math.sqrt(3)

def construct():
    t0 = time.time()
    
    # Precompute unit hexagon vertices (angle 0)
    # Vertices at 0, 60, 120, 180, 240, 300 degrees
    angles = np.radians(np.arange(0, 360, 60))
    unit_hex_verts = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Precompute outward normals for unit hexagon (perpendicular to edges)
    # Edge i goes from vertex i to vertex (i+1)%6
    # Normal points outward
    unit_hex_normals = np.zeros((6, 2))
    for i in range(6):
        ex = unit_hex_verts[(i+1)%6, 0] - unit_hex_verts[i, 0]
        ey = unit_hex_verts[(i+1)%6, 1] - unit_hex_verts[i, 1]
        # Outward normal: rotate edge vector by -90 degrees (clockwise)
        unit_hex_normals[i, 0] = ey
        unit_hex_normals[i, 1] = -ex
        norm = math.sqrt(unit_hex_normals[i, 0]**2 + unit_hex_normals[i, 1]**2)
        if norm > 1e-10:
            unit_hex_normals[i, 0] /= norm
            unit_hex_normals[i, 1] /= norm
    
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

    # Helper: Check if two hexagons are disjoint using SAT with precomputed normals
    # h1, h2 are (x, y, angle_rad)
    def hexagons_disjoint_fast(h1, h2):
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        
        # If far apart, disjoint (max distance between centers for touching is 2*sqrt(3) for flat-to-flat)
        if dist_sq > 12.0: # dist > sqrt(12) ~ 3.46
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
        
        # SAT using precomputed normals rotated by each hexagon's angle
        # For hexagon 1: normal_i rotated by a1
        # For hexagon 2: normal_i rotated by a2
        
        # Check all 12 potential separating axes
        for i in range(6):
            # Normal from hex1
            nx = unit_hex_normals[i, 0] * c1 - unit_hex_normals[i, 1] * s1
            ny = unit_hex_normals[i, 0] * s1 + unit_hex_normals[i, 1] * c1
            
            proj1 = v1x * nx + v1y * ny
            proj2 = v2x * nx + v2y * ny
            
            if proj1.max() < proj2.min() or proj2.max() < proj1.min():
                return True
        
        for i in range(6):
            # Normal from hex2
            nx = unit_hex_normals[i, 0] * c2 - unit_hex_normals[i, 1] * s2
            ny = unit_hex_normals[i, 0] * s2 + unit_hex_normals[i, 1] * c2
            
            proj1 = v1x * nx + v1y * ny
            proj2 = v2x * nx + v2y * ny
            
            if proj1.max() < proj2.min() or proj2.max() < proj1.min():
                return True
                
        return False

    def compute_min_outer_side(positions, angles_deg):
        """
        Compute the minimum side length of a regular hexagon containing all inner hexagons.
        Returns (side_length, optimal_outer_angle_deg, center_x, center_y)
        """
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        
        # Flatten to (72, 2)
        verts_flat = all_verts.reshape(-1, 2)
        
        # Center
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        best_side = float('inf')
        best_phi = 0.0
        
        # Coarse scan
        for phi_deg in range(0, 60, 2):
            phi = np.radians(phi_deg)
            max_extent = 0.0
            for k in range(6):
                normal_angle = phi + np.pi/6 + k * np.pi/3
                nx = math.cos(normal_angle)
                ny = math.sin(normal_angle)
                
                proj = offsets[:, 0] * nx + offsets[:, 1] * ny
                max_proj = np.max(np.abs(proj))
                
                side_k = (2.0 / SQRT3) * max_proj
                if side_k > max_extent:
                    max_extent = side_k
            
            if max_extent < best_side:
                best_side = max_extent
                best_phi = phi_deg
                
        # Fine scan
        for phi_deg in np.arange(best_phi - 2, best_phi + 2, 0.1):
            if phi_deg < 0 or phi_deg >= 60: continue
            phi = np.radians(phi_deg)
            max_extent = 0.0
            for k in range(6):
                normal_angle = phi + np.pi/6 + k * np.pi/3
                nx = math.cos(normal_angle)
                ny = math.sin(normal_angle)
                
                proj = offsets[:, 0] * nx + offsets[:, 1] * ny
                max_proj = np.max(np.abs(proj))
                
                side_k = (2.0 / SQRT3) * max_proj
                if side_k > max_extent:
                    max_extent = side_k
            
            if max_extent < best_side:
                best_side = max_extent
                best_phi = phi_deg

        return best_side, best_phi, center[0], center[1]

    def is_valid_configuration(positions, angles_deg):
        """Check if all hexagons are pairwise disjoint."""
        # Convert angles to radians for disjoint check
        angles_rad = np.radians(angles_deg)
        for i in range(N):
            for j in range(i+1, N):
                h1 = (positions[i][0], positions[i][1], angles_rad[i])
                h2 = (positions[j][0], positions[j][1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    return False
        return True

    # Initial Configuration: Honeycomb-like with proper spacing
    # For unit hexagons, center-to-center distance for touching is 2*sqrt(3) ≈ 3.464
    # Honeycomb pattern: alternate rows offset by half the horizontal spacing
    cols = 4
    rows = 3
    h_spacing = 3.0  # horizontal spacing between centers in same row
    v_spacing = SQRT3  # vertical spacing between rows
    
    initial_positions = np.zeros((N, 2))
    initial_angles = np.zeros(N)
    
    idx = 0
    for r in range(rows):
        for c in range(cols):
            # Offset alternate rows by half the horizontal spacing
            offset = (h_spacing / 2.0) if (r % 2 == 1) else 0.0
            initial_positions[idx, 0] = -1.5 * h_spacing + c * h_spacing + offset
            initial_positions[idx, 1] = -(rows - 1) / 2.0 * v_spacing + r * v_spacing
            initial_angles[idx] = 0.0
            idx += 1
            
    best_side = float('inf')
    best_positions = initial_positions.copy()
    best_angles = initial_angles.copy()
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0
    
    current_positions = initial_positions.copy()
    current_angles = initial_angles.copy()
    current_side, current_phi, current_cx, current_cy = compute_min_outer_side(current_positions, current_angles)
    
    if current_side < best_side:
        best_side = current_side
        best_positions = current_positions.copy()
        best_angles = current_angles.copy()
        best_phi = current_phi
        best_cx = current_cx
        best_cy = current_cy

    T = 1.0
    T_min = 0.001
    cooling = 0.9995
    step_size = 0.5
    step_size_min = 0.1
    step_size_max = 1.5
    
    step_count = 0
    max_steps = 50000
    consecutive_rejects = 0
    
    # SA Loop
    while time.time() - t0 < DEADLINE * 0.8 and step_count < max_steps:
        step_count += 1
        
        idx = random.randint(0, N - 1)
        new_positions = current_positions.copy()
        new_angles = current_angles.copy()
        
        dx = random.gauss(0, step_size)
        dy = random.gauss(0, step_size)
        da = random.gauss(0, 10.0) # degrees
        
        new_positions[idx, 0] += dx
        new_positions[idx, 1] += dy
        new_angles[idx] += da
        
        # Check validity
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
            consecutive_rejects += 1
            # Adapt step size: reduce if too many rejections
            if consecutive_rejects > 100:
                step_size = max(step_size_min, step_size * 0.95)
                consecutive_rejects = 0
            T *= cooling
            if T < T_min: T = T_min
            continue
            
        consecutive_rejects = 0
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
            # Increase step size slightly on success
            step_size = min(step_size_max, step_size * 1.02)
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

    # Polish Step
    # Use Nelder-Mead on the best configuration found
    # Variables: 36 (x, y, angle for each hex)
    def polish_objective(params):
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(N):
            pos[i, 0] = params[3*i]
            pos[i, 1] = params[3*i+1]
            ang[i] = params[3*i+2]
            
        # Check validity with penalty
        angles_rad = np.radians(ang)
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                h1 = (pos[i, 0], pos[i, 1], angles_rad[i])
                h2 = (pos[j, 0], pos[j, 1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    penalty += 10.0
        
        side, phi, cx, cy = compute_min_outer_side(pos, ang)
        return side + penalty

    # Initial params for polish
    init_params = np.zeros(3*N)
    for i in range(N):
        init_params[3*i] = best_positions[i, 0]
        init_params[3*i+1] = best_positions[i, 1]
        init_params[3*i+2] = best_angles[i]
        
    # Run minimize with tighter tolerances
    if time.time() - t0 < DEADLINE * 0.95:
        try:
            res = minimize(polish_objective, init_params, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if res.fun < best_side + 1.0:  # Allow small penalty for improvement check
                # Update best
                new_pos = np.zeros((N, 2))
                new_ang = np.zeros(N)
                for i in range(N):
                    new_pos[i, 0] = res.x[3*i]
                    new_pos[i, 1] = res.x[3*i+1]
                    new_ang[i] = res.x[3*i+2]
                    
                # Verify validity one last time
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
            pass # Ignore errors in polish, keep SA result

    # Final Result
    # Construct inner list: (x, y, angle_degrees)
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle

# EVOLVE-BLOCK-END
