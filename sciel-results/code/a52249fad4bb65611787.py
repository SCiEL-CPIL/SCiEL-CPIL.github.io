import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0  # Safety margin before 150s wall-clock kill

def construct():
    t0 = time.time()
    
    # Precompute unit hexagon vertices (angle 0)
    angles = np.radians(np.arange(0, 360, 60))
    unit_hex_verts = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Precompute edge normals for unit hexagon (6 normals)
    # For a regular hexagon with vertices at angles 0, 60, ..., 300,
    # the outward normal to edge i (from vertex i to i+1) points at angle (i*60 + 30)
    normal_angles = np.radians(np.arange(0, 360, 60) + 30)
    edge_normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))

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
        
        # If distance > 2 (circumradius sum), they definitely don't overlap
        if dist_sq > 4.0:
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
        for i in range(6):
            nx1 = edge_normals[i, 0] * c1 - edge_normals[i, 1] * s1
            ny1 = edge_normals[i, 0] * s1 + edge_normals[i, 1] * c1
            
            proj1 = v1x * nx1 + v1y * ny1
            proj2 = v2x * nx1 + v2y * ny1
            
            if np.max(proj1) < np.min(proj2) or np.max(proj2) < np.min(proj1):
                return True
                
        for i in range(6):
            nx2 = edge_normals[i, 0] * c2 - edge_normals[i, 1] * s2
            ny2 = edge_normals[i, 0] * s2 + edge_normals[i, 1] * c2
            
            proj1 = v1x * nx2 + v1y * ny2
            proj2 = v2x * nx2 + v2y * ny2
            
            if np.max(proj1) < np.min(proj2) or np.max(proj2) < np.min(proj1):
                return True
                
        return False

    def compute_min_outer_side_fast(positions, angles_deg):
        """
        Compute the minimum side length of a regular hexagon containing all inner hexagons.
        Uses Golden Section Search for the optimal outer hexagon angle.
        Returns (side_length, optimal_outer_angle_deg, center_x, center_y)
        """
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        sqrt3 = math.sqrt(3)
        
        def side_for_phi(phi_rad):
            normal_angles = phi_rad + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            max_abs_proj = np.max(np.abs(proj), axis=0)
            max_extent = np.max(max_abs_proj)
            
            return (2.0 / sqrt3) * max_extent
        
        # Golden Section Search in [0, 60) degrees
        lo = 0.0
        hi = 60.0
        inv_phi = (1 + math.sqrt(5)) / 2
        phi1 = hi - (hi - lo) / inv_phi
        phi2 = lo + (hi - lo) / inv_phi
        
        f1 = side_for_phi(np.radians(phi1))
        f2 = side_for_phi(np.radians(phi2))
        
        for _ in range(40):
            if f1 < f2:
                hi = phi2
                phi2 = phi1
                f2 = f1
                phi1 = hi - (hi - lo) / inv_phi
                f1 = side_for_phi(np.radians(phi1))
            else:
                lo = phi1
                phi1 = phi2
                f1 = f2
                phi2 = lo + (hi - lo) / inv_phi
                f2 = side_for_phi(np.radians(phi2))
                
        best_phi = (lo + hi) / 2.0
        best_side = side_for_phi(np.radians(best_phi))
        
        return best_side, best_phi, center[0], center[1]

    def smooth_side_objective(params, beta=200.0):
        """
        Smooth surrogate for the outer hexagon side length using log-sum-exp.
        params: (3*N,) array with [x0,y0,a0, x1,y1,a1, ...]
        Returns a smooth approximation of the minimum containing hexagon side.
        """
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(N):
            pos[i, 0] = params[3*i]
            pos[i, 1] = params[3*i+1]
            ang[i] = params[3*i+2]
        
        angles_rad = np.radians(ang)
        all_verts = get_all_vertices(pos, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        sqrt3 = math.sqrt(3)
        
        # We optimize over 6 normal directions simultaneously using a smooth approach.
        # The side length for a given orientation phi is:
        # s(phi) = (2/sqrt3) * max over 6 directions of max_abs_proj(direction)
        # 
        # To make this smooth, we use log-sum-exp to approximate the max.
        # But we also need to optimize phi. Let's parameterize phi in [0, 60) 
        # and use a smooth max over the 6 face normals.
        
        # For a smooth optimization, we can fix the center at the centroid
        # and optimize the 6 support values. But that's complex.
        # Instead, let's use a different smooth approach:
        # For each of the 6 normal directions of the outer hexagon (at angles phi+30, phi+90, ...),
        # the required apothem is max(|proj|). 
        # The side is (2/sqrt3) * max(apothems).
        # 
        # We can smooth the outer max over the 6 directions using log-sum-exp.
        # And the inner max(|proj|) can be smoothed similarly.
        
        # Actually, for the polish step, let's just use the exact function but with
        # a smooth penalty for overlaps. The main benefit of smoothness is in
        # handling the collision constraints.
        
        # For the objective, let's compute the exact side but add a smooth
        # penalty for any pair that is too close (using a soft min distance).
        
        # Compute exact side
        def side_for_phi(phi_rad):
            normal_angles = phi_rad + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            max_abs_proj = np.max(np.abs(proj), axis=0)
            max_extent = np.max(max_abs_proj)
            
            return (2.0 / sqrt3) * max_extent
        
        # Golden section for phi
        lo = 0.0
        hi = 60.0
        inv_phi = (1 + math.sqrt(5)) / 2
        phi1 = hi - (hi - lo) / inv_phi
        phi2 = lo + (hi - lo) / inv_phi
        
        f1 = side_for_phi(np.radians(phi1))
        f2 = side_for_phi(np.radians(phi2))
        
        for _ in range(30):
            if f1 < f2:
                hi = phi2
                phi2 = phi1
                f2 = f1
                phi1 = hi - (hi - lo) / inv_phi
                f1 = side_for_phi(np.radians(phi1))
            else:
                lo = phi1
                phi1 = phi2
                f1 = f2
                phi2 = lo + (hi - lo) / inv_phi
                f2 = side_for_phi(np.radians(phi2))
                
        best_phi = (lo + hi) / 2.0
        best_side = side_for_phi(np.radians(best_phi))
        
        # Add smooth collision penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i,0] - pos[j,0]
                dy = pos[i,1] - pos[j,1]
                dist = math.sqrt(dx*dx + dy*dy)
                # Minimum center distance for non-overlap is sqrt(3)
                # Use a smooth penalty: if dist < sqrt(3), add penalty
                if dist < math.sqrt(3):
                    # Smooth quadratic penalty
                    deficit = math.sqrt(3) - dist
                    penalty += beta * deficit * deficit
        
        return best_side + penalty

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

    # Initial Configuration: Honeycomb
    # A honeycomb of 12 hexagons can be arranged in 3 rows of 4.
    # Distance between centers of adjacent hexes is sqrt(3).
    # Vertical distance between rows is sqrt(3)/2.
    # Horizontal distance within a row is 1.5 (for offset rows) or 3.0 (for same row neighbors in a staggered grid)?
    # Actually, in a standard honeycomb:
    # Row 0: x = -4.5, -1.5, 1.5, 4.5 (step 3.0)
    # Row 1: x = -3.0, 0.0, 3.0 (step 3.0, offset 1.5) -> Wait, 4 hexes in row 1?
    # If we have 3 rows of 4:
    # Row 0: 4 hexes
    # Row 1: 4 hexes
    # Row 2: 4 hexes
    # This is a rectangular block of hexagons.
    # Let's use the coordinates that form a compact hexagonal cluster.
    # A hexagonal number is 1, 7, 19. 12 is not a hexagonal number.
    # So we can't form a perfect hexagon.
    # Let's try a 3x4 rectangular honeycomb.
    # Row 0: y=0, x = -4.5, -1.5, 1.5, 4.5
    # Row 1: y=sqrt(3)/2, x = -3.0, 0.0, 3.0, 6.0? No, that's not symmetric.
    # Let's just generate a valid honeycomb pattern.
    
    pos_init = np.zeros((N, 2))
    ang_init = np.zeros(N)
    
    sqrt3 = math.sqrt(3)
    v_step = sqrt3 / 2.0
    h_step = 1.5
    
    # Let's try a 2x6 grid? Or 3x4?
    # 3 rows of 4 is likely more compact.
    # Row 0: 4 hexes
    # Row 1: 4 hexes (offset)
    # Row 2: 4 hexes
    # Let's place them.
    idx = 0
    for r in range(3):
        for c in range(4):
            x = (c - 1.5) * 3.0 + (r % 2) * 1.5
            y = r * v_step * 2.0 # Wait, vertical step is sqrt(3)/2.
            # Let's re-verify.
            # Distance between (0,0) and (1.5, y) should be sqrt(3).
            # 1.5^2 + y^2 = 3 => 2.25 + y^2 = 3 => y^2 = 0.75 => y = sqrt(3)/2.
            # So y_step = sqrt(3)/2.
            y = r * (sqrt3 / 2.0)
            
            pos_init[idx, 0] = x
            pos_init[idx, 1] = y
            ang_init[idx] = 0.0
            idx += 1
            
    # Check validity of init
    if not is_valid_configuration(pos_init, ang_init):
        # Fallback to grid
        pos_init = np.zeros((N, 2))
        ang_init = np.zeros(N)
        idx = 0
        for r in range(3):
            for c in range(4):
                pos_init[idx, 0] = (c - 1.5) * 3.0
                pos_init[idx, 1] = r * sqrt3
                idx += 1

    best_side, best_phi, best_cx, best_cy = compute_min_outer_side_fast(pos_init, ang_init)
    
    current_positions = pos_init.copy()
    current_angles = ang_init.copy()
    current_side = best_side
    
    best_positions = current_positions.copy()
    best_angles = current_angles.copy()
    best_phi_final = best_phi
    best_cx_final = best_cx
    best_cy_final = best_cy

    T = 0.5
    T_min = 0.0005
    cooling = 0.9995
    step_size = 0.3
    
    step_count = 0
    max_steps = 50000
    
    # SA Loop
    while time.time() - t0 < DEADLINE * 0.7 and step_count < max_steps:
        step_count += 1
        
        idx = random.randint(0, N - 1)
        new_positions = current_positions.copy()
        new_angles = current_angles.copy()
        
        dx = random.gauss(0, step_size)
        dy = random.gauss(0, step_size)
        da = random.gauss(0, 10.0)  # degrees
        
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
            T *= cooling
            if T < T_min: T = T_min
            continue
            
        new_side, new_phi, new_cx, new_cy = compute_min_outer_side_fast(new_positions, new_angles)
        
        if new_side < current_side:
            current_positions = new_positions
            current_angles = new_angles
            current_side = new_side
            
            if new_side < best_side:
                best_side = new_side
                best_positions = new_positions.copy()
                best_angles = new_angles.copy()
                best_phi_final = new_phi
                best_cx_final = new_cx
                best_cy_final = new_cy
        else:
            delta = new_side - current_side
            if delta < 0: delta = 0
            if random.random() < math.exp(-delta / T):
                current_positions = new_positions
                current_angles = new_angles
                current_side = new_side
                
                if new_side < best_side:
                    best_side = new_side
                    best_positions = new_positions.copy()
                    best_angles = new_angles.copy()
                    best_phi_final = new_phi
                    best_cx_final = new_cx
                    best_cy_final = new_cy
                    
        T *= cooling
        if T < T_min: T = T_min

    # Polish Step with smooth surrogate
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
        
        side, phi, cx, cy = compute_min_outer_side_fast(pos, ang)
        return side

    def smooth_polish_objective(params, beta=100.0):
        """Smooth version using soft collision penalty."""
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(N):
            pos[i, 0] = params[3*i]
            pos[i, 1] = params[3*i+1]
            ang[i] = params[3*i+2]
        
        angles_rad = np.radians(ang)
        
        # Smooth collision penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i,0] - pos[j,0]
                dy = pos[i,1] - pos[j,1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = 3.0  # sqrt(3)^2
                if dist_sq < min_dist_sq:
                    dist = math.sqrt(dist_sq)
                    deficit = math.sqrt(min_dist_sq) - dist
                    penalty += beta * deficit * deficit
        
        if penalty > 100.0:
            return penalty
        
        side, phi, cx, cy = compute_min_outer_side_fast(pos, ang)
        return side + penalty * 0.1

    init_params = np.zeros(3*N)
    for i in range(N):
        init_params[3*i] = best_positions[i, 0]
        init_params[3*i+1] = best_positions[i, 1]
        init_params[3*i+2] = best_angles[i]
        
    polish_results = []
    for attempt in range(8):
        if time.time() - t0 > DEADLINE * 0.94:
            break
        
        try:
            if attempt == 0:
                p = init_params.copy()
            else:
                p = init_params + random.gauss(0, 0.15, 3*N)
            
            # Use smooth objective for gradient-friendly optimization
            res = minimize(smooth_polish_objective, p, method='Nelder-Mead', 
                           options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-9})
            
            if res.fun < 1e5:
                new_pos = np.zeros((N, 2))
                new_ang = np.zeros(N)
                for i in range(N):
                    new_pos[i, 0] = res.x[3*i]
                    new_pos[i, 1] = res.x[3*i+1]
                    new_ang[i] = res.x[3*i+2]
                
                if is_valid_configuration(new_pos, new_ang):
                    side, phi, cx, cy = compute_min_outer_side_fast(new_pos, new_ang)
                    polish_results.append((side, new_pos, new_ang, phi, cx, cy))
                    
                    # If this is better, update init for next attempts
                    if side < best_side:
                        best_side = side
                        best_positions = new_pos.copy()
                        best_angles = new_ang.copy()
                        best_phi_final = phi
                        best_cx_final = cx
                        best_cy_final = cy
                        for i in range(N):
                            init_params[3*i] = new_pos[i, 0]
                            init_params[3*i+1] = new_pos[i, 1]
                            init_params[3*i+2] = new_ang[i]
        except Exception:
            pass
    
    if polish_results:
        polish_results.sort(key=lambda x: x[0])
        best_side, best_positions, best_angles, best_phi_final, best_cx_final, best_cy_final = polish_results[0]
    
    # Final high-precision polish with exact objective
    if time.time() - t0 < DEADLINE * 0.97:
        try:
            res = minimize(polish_objective, init_params, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if res.fun < 1e5:
                new_pos = np.zeros((N, 2))
                new_ang = np.zeros(N)
                for i in range(N):
                    new_pos[i, 0] = res.x[3*i]
                    new_pos[i, 1] = res.x[3*i+1]
                    new_ang[i] = res.x[3*i+2]
                
                if is_valid_configuration(new_pos, new_ang):
                    side, phi, cx, cy = compute_min_outer_side_fast(new_pos, new_ang)
                    if side < best_side:
                        best_side = side
                        best_positions = new_pos.copy()
                        best_angles = new_ang.copy()
                        best_phi_final = phi
                        best_cx_final = cx
                        best_cy_final = cy
        except Exception:
            pass

    # Final Result
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx_final), float(best_cy_final))
    side = float(best_side)
    angle = float(best_phi_final)
    
    return inner, center, side, angle
