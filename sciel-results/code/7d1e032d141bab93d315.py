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
    # The normal to edge i is perpendicular to the edge vector.
    # For a regular hexagon, the outward normal to edge i (connecting vertex i and i+1)
    # is at angle i*60 + 30 degrees? 
    # Vertex i is at angle i*60. Edge i is from i*60 to (i+1)*60.
    # Midpoint angle is i*60 + 30. Normal is radial.
    # So normal i is at angle (i*60 + 30) degrees.
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
        
        # Max distance for overlap is 2 * circumradius = 2.
        # If dist > 2, they are disjoint? No, if they are far apart.
        # If dist > 2 * sqrt(3) (apothem * 2 + ...), definitely disjoint.
        # Actually, if dist > 2, they might still overlap if rotated?
        # The max distance between centers for touching hexagons is 2 (if vertices touch) 
        # or slightly more? No, max distance for overlap is 2 (vertex to vertex).
        # If dist > 2, they are disjoint.
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
        v1 = np.column_stack((v1x, v1y))
        
        v2x = h2[0] + ux * c2 - uy * s2
        v2y = h2[1] + ux * s2 + uy * c2
        v2 = np.column_stack((v2x, v2y))
        
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
            # Normals of the outer hexagon
            # The outer hexagon has normals at phi + 30, phi + 90, ...
            # The width in direction normal n is max(proj) - min(proj)
            # Side length s = width / sqrt(3) ?
            # For a regular hexagon, width = s * sqrt(3). So s = width / sqrt(3).
            # The "width" is the projection onto the normal.
            # We need the max absolute projection from the center?
            # If the hexagon is centered at C, the vertices are at distance s from C? No.
            # The apothem is s * sqrt(3) / 2.
            # The max projection onto a normal is the apothem + max offset?
            # Actually, if we center the outer hexagon at the centroid of the points,
            # the required apothem is max(|proj - center_proj|).
            # Here offsets are centered at 0. So max_abs_proj is the required apothem.
            # s = 2 * apothem / sqrt(3).
            
            normal_angles = phi_rad + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            max_abs_proj = np.max(np.abs(proj), axis=0)
            max_extent = np.max(max_abs_proj)
            
            return (2.0 / sqrt3) * max_extent

        # Golden Section Search in [0, 60) degrees
        # The function is periodic with 60 deg.
        lo = 0.0
        hi = 60.0
        phi1 = hi - (hi - lo) / ((1 + math.sqrt(5)) / 2)
        phi2 = lo + (hi - lo) / ((1 + math.sqrt(5)) / 2)
        
        f1 = side_for_phi(np.radians(phi1))
        f2 = side_for_phi(np.radians(phi2))
        
        for _ in range(40): # ~40 iterations gives high precision
            if f1 < f2:
                hi = phi2
                phi2 = phi1
                f2 = f1
                phi1 = hi - (hi - lo) / ((1 + math.sqrt(5)) / 2)
                f1 = side_for_phi(np.radians(phi1))
            else:
                lo = phi1
                phi1 = phi2
                f1 = f2
                phi2 = lo + (hi - lo) / ((1 + math.sqrt(5)) / 2)
                f2 = side_for_phi(np.radians(phi2))
                
        best_phi = (lo + hi) / 2.0
        best_side = side_for_phi(np.radians(best_phi))
        
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

    # Initial Configuration: Honeycomb
    # A honeycomb of 12 hexagons can be arranged in 3 rows of 4, or 2 rows of 6?
    # Let's try a compact cluster.
    # Center at (0,0).
    # Hex 0 at (0,0)
    # Neighbors at distance sqrt(3) (apothem*2? No, distance between centers in honeycomb is sqrt(3) * side? 
    # Side=1. Distance between centers in honeycomb is sqrt(3).
    # Let's generate a valid honeycomb.
    
    pos_init = np.zeros((N, 2))
    ang_init = np.zeros(N)
    
    # Simple 3x4 grid is not honeycomb.
    # Honeycomb:
    # Row 0: 4 hexes
    # Row 1: 4 hexes (offset)
    # Row 2: 4 hexes
    # This is 12 hexes.
    # Distance between rows is 1.5 * sqrt(3)? No.
    # Vertical distance between centers in honeycomb: sqrt(3) * 1 = sqrt(3) ?
    # Let's check.
    # Hex side 1. Height = sqrt(3).
    # If we stack them, vertical distance is sqrt(3) * 0.75? No.
    # In a honeycomb, the vertical distance between rows is sqrt(3) * 1 * 0.75? 
    # Actually, the distance between centers of adjacent hexes is sqrt(3).
    # If we have a row, the next row is offset by sqrt(3)/2 horizontally and sqrt(3)*sqrt(3)/2 = 1.5 vertically?
    # Let's use the known honeycomb coordinates.
    # Centers: (i * 1.5 + j * 0.75, j * sqrt(3) * 0.75 * 2?)
    
    # Let's just use a simple valid config: 2 rows of 6.
    # This might be wider than a 3x4 honeycomb.
    # Let's try 3 rows of 4.
    # Row 0: x = -4.5, -1.5, 1.5, 4.5 (step 3.0)
    # Row 1: x = -3.0, 0.0, 3.0 (step 3.0, offset 1.5)
    # Row 2: x = -4.5, -1.5, 1.5, 4.5
    # Vertical step: sqrt(3) * 1.0? No.
    # Distance between (0,0) and (1.5, y) should be sqrt(3).
    # 1.5^2 + y^2 = 3 => 2.25 + y^2 = 3 => y^2 = 0.75 => y = sqrt(3)/2.
    # So vertical step is sqrt(3)/2.
    
    sqrt3 = math.sqrt(3)
    v_step = sqrt3 / 2.0
    h_step = 1.5
    
    idx = 0
    for r in range(3):
        for c in range(4):
            x = (c - 1.5) * 3.0 + (r % 2) * 1.5
            y = r * v_step * 2.0 # Wait, if r=0, y=0. r=1, y=sqrt(3). r=2, y=2*sqrt(3).
            # Let's re-verify.
            # Row 0: y=0
            # Row 1: y=sqrt(3)/2 * 2? No.
            # Distance between (x, 0) and (x+1.5, y) is sqrt(3).
            # 1.5^2 + y^2 = 3 => y = sqrt(0.75) = sqrt(3)/2.
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
        
        side, phi, cx, cy = compute_min_outer_side_fast(pos, ang)
        return side

    init_params = np.zeros(3*N)
    for i in range(N):
        init_params[3*i] = best_positions[i, 0]
        init_params[3*i+1] = best_positions[i, 1]
        init_params[3*i+2] = best_angles[i]
        
    polish_results = []
    for attempt in range(5):
        if time.time() - t0 > DEADLINE * 0.92:
            break
        
        try:
            if attempt == 0:
                p = init_params.copy()
            else:
                p = init_params + random.gauss(0, 0.2, 3*N)
            
            res = minimize(polish_objective, p, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
            
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
        except Exception:
            pass
    
    if polish_results:
        polish_results.sort(key=lambda x: x[0])
        best_side, best_positions, best_angles, best_phi_final, best_cx_final, best_cy_final = polish_results[0]

    # Final Result
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx_final), float(best_cy_final))
    side = float(best_side)
    angle = float(best_phi_final)
    
    return inner, center, side, angle

# EVOLVE-BLOCK-END
