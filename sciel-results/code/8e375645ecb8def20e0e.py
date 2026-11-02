import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0
SQRT3 = math.sqrt(3)

def construct():
    t0 = time.time()
    
    # Precompute unit hexagon vertices (angle 0)
    angles = np.radians(np.arange(0, 360, 60))
    unit_hex_verts = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Precompute outward normals for unit hexagon edges
    unit_hex_normals = np.zeros((6, 2))
    for i in range(6):
        ex = unit_hex_verts[(i+1)%6, 0] - unit_hex_verts[i, 0]
        ey = unit_hex_verts[(i+1)%6, 1] - unit_hex_verts[i, 1]
        unit_hex_normals[i, 0] = ey
        unit_hex_normals[i, 1] = -ex
        norm = math.sqrt(unit_hex_normals[i, 0]**2 + unit_hex_normals[i, 1]**2)
        if norm > 1e-9:
            unit_hex_normals[i] /= norm
    
    # Helper: Get vertices of all hexagons given positions and angles
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
    
    # Helper: Check if two hexagons are disjoint using SAT
    def hexagons_disjoint_fast(h1, h2):
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        
        if dist_sq > 12.0:
            return True
        
        a1 = h1[2]
        a2 = h2[2]
        
        c1, s1 = math.cos(a1), math.sin(a1)
        c2, s2 = math.cos(a2), math.sin(a2)
        
        ux = unit_hex_verts[:, 0]
        uy = unit_hex_verts[:, 1]
        
        n1x = unit_hex_normals[:, 0] * c1 - unit_hex_normals[:, 1] * s1
        n1y = unit_hex_normals[:, 0] * s1 + unit_hex_normals[:, 1] * c1
        n2x = unit_hex_normals[:, 0] * c2 - unit_hex_normals[:, 1] * s2
        n2y = unit_hex_normals[:, 0] * s2 + unit_hex_normals[:, 1] * c2
        
        v1x = h1[0] + ux * c1 - uy * s1
        v1y = h1[1] + ux * s1 + uy * c1
        v2x = h2[0] + ux * c2 - uy * s2
        v2y = h2[1] + ux * s2 + uy * c2
        
        for i in range(6):
            nx, ny = n1x[i], n1y[i]
            proj1_min = np.min(v1x * nx + v1y * ny)
            proj1_max = np.max(v1x * nx + v1y * ny)
            proj2_min = np.min(v2x * nx + v2y * ny)
            proj2_max = np.max(v2x * nx + v2y * ny)
            
            if proj1_max < proj2_min or proj2_max < proj1_min:
                return True
        
        for i in range(6):
            nx, ny = n2x[i], n2y[i]
            proj1_min = np.min(v1x * nx + v1y * ny)
            proj1_max = np.max(v1x * nx + v1y * ny)
            proj2_min = np.min(v2x * nx + v2y * ny)
            proj2_max = np.max(v2x * nx + v2y * ny)
            
            if proj1_max < proj2_min or proj2_max < proj1_min:
                return True
        
        return False
    
    def compute_min_outer_side(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        best_side = float('inf')
        best_phi = 0.0
        
        def side_for_phi(phi_deg):
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
            return max_extent
        
        for phi_deg in range(0, 60, 2):
            s = side_for_phi(phi_deg)
            if s < best_side:
                best_side = s
                best_phi = phi_deg
        
        for phi_deg in np.arange(best_phi - 2, best_phi + 2, 0.1):
            if phi_deg < 0 or phi_deg >= 60:
                continue
            s = side_for_phi(phi_deg)
            if s < best_side:
                best_side = s
                best_phi = phi_deg
        
        for phi_deg in np.arange(best_phi - 0.1, best_phi + 0.1, 0.01):
            if phi_deg < 0 or phi_deg >= 60:
                continue
            s = side_for_phi(phi_deg)
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
    
    # Make honeycomb initialization - this is the key structural change
    def make_honeycomb_init(seed_offset=0):
        """
        Create a dense honeycomb cluster of 12 hexagons.
        The honeycomb lattice has hexagons touching on all 6 sides.
        We create a compact cluster that minimizes the bounding hexagon.
        """
        rng = np.random.RandomState(seed_offset)
        
        # Honeycomb lattice vectors
        # For unit hexagons, the lattice spacing is:
        # Horizontal: 3 (distance between centers of adjacent hexagons in same row)
        # Vertical: sqrt(3) (distance between rows)
        # Offset for alternating rows: 1.5 horizontally
        
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        
        # Create a 4x3 grid in honeycomb pattern
        # Row 0: 4 hexagons
        # Row 1: 3 hexagons (offset by 1.5)
        # Row 2: 4 hexagons
        # Row 3: 1 hexagon (to make 12)
        
        idx = 0
        # Row 0
        for c in range(4):
            positions[idx, 0] = (c - 1.5) * 3.0
            positions[idx, 1] = 0.0
            angles[idx] = 0.0
            idx += 1
        
        # Row 1 (offset)
        for c in range(3):
            positions[idx, 0] = (c - 1.0) * 3.0 + 1.5
            positions[idx, 1] = SQRT3
            angles[idx] = 0.0
            idx += 1
        
        # Row 2
        for c in range(4):
            positions[idx, 0] = (c - 1.5) * 3.0
            positions[idx, 1] = 2 * SQRT3
            angles[idx] = 0.0
            idx += 1
        
        # Row 3
        positions[idx, 0] = 0.0
        positions[idx, 1] = 3 * SQRT3
        angles[idx] = 0.0
        idx += 1
        
        # Center the configuration
        center = np.mean(positions, axis=0)
        positions -= center
        
        # Add small random perturbation
        positions += rng.normal(0, 0.05, size=(N, 2))
        angles += rng.normal(0, 2.0, size=N)
        
        return positions, angles
    
    # Try multiple initial configurations
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0
    
    num_inits = 8
    for init_i in range(num_inits):
        init_positions, init_angles = make_honeycomb_init(init_i * 100)
        
        side, phi, cx, cy = compute_min_outer_side(init_positions, init_angles)
        
        if side < best_side:
            best_side = side
            best_positions = init_positions.copy()
            best_angles = init_angles.copy()
            best_phi = phi
            best_cx = cx
            best_cy = cy
    
    # Dual-Phase Rigid-Body Relaxation
    # Phase 1: Global rigid-body motion optimization
    # Phase 2: Local relaxation with SAT constraints
    
    def rigid_body_objective(params):
        """
        Optimize global rigid-body motion: translation (tx, ty) and rotation (theta)
        of the entire cluster, plus individual hexagon rotations.
        """
        tx = params[0]
        ty = params[1]
        theta = params[2]
        
        # Apply rigid body transform to positions
        c, s = math.cos(theta), math.sin(theta)
        new_positions = np.zeros((N, 2))
        for i in range(N):
            x, y = best_positions[i, 0], best_positions[i, 1]
            new_positions[i, 0] = x * c - y * s + tx
            new_positions[i, 1] = x * s + y * c + ty
        
        # Individual rotations
        new_angles = best_angles + params[3:3+N]
        
        # Check validity
        angles_rad = np.radians(new_angles)
        for i in range(N):
            for j in range(i+1, N):
                h1 = (new_positions[i, 0], new_positions[i, 1], angles_rad[i])
                h2 = (new_positions[j, 0], new_positions[j, 1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    return 1e6
        
        side, phi, cx, cy = compute_min_outer_side(new_positions, new_angles)
        return side
    
    # Phase 1: Rigid body optimization
    if time.time() - t0 < DEADLINE * 0.5:
        init_params = np.zeros(3 + N)
        # Start with small perturbations
        rng = np.random.RandomState(42)
        init_params[0] = rng.normal(0, 0.1)
        init_params[1] = rng.normal(0, 0.1)
        init_params[2] = rng.normal(0, 5.0) * np.pi / 180.0
        init_params[3:3+N] = rng.normal(0, 1.0, size=N) * np.pi / 180.0
        
        try:
            res = minimize(rigid_body_objective, init_params, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-7})
            
            if res.fun < 1e5:
                tx, ty, theta = res.x[0], res.x[1], res.x[2]
                c, s = math.cos(theta), math.sin(theta)
                new_positions = np.zeros((N, 2))
                for i in range(N):
                    x, y = best_positions[i, 0], best_positions[i, 1]
                    new_positions[i, 0] = x * c - y * s + tx
                    new_positions[i, 1] = x * s + y * c + ty
                
                new_angles = best_angles + res.x[3:3+N] * 180.0 / np.pi
                
                # Verify
                angles_rad = np.radians(new_angles)
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        h1 = (new_positions[i, 0], new_positions[i, 1], angles_rad[i])
                        h2 = (new_positions[j, 0], new_positions[j, 1], angles_rad[j])
                        if not hexagons_disjoint_fast(h1, h2):
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    side, phi, cx, cy = compute_min_outer_side(new_positions, new_angles)
                    if side < best_side:
                        best_side = side
                        best_positions = new_positions
                        best_angles = new_angles
                        best_phi = phi
                        best_cx = cx
                        best_cy = cy
        except Exception:
            pass
    
    # Phase 2: Local relaxation with individual hexagon adjustments
    def local_relax_objective(params):
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
    
    def polish_from(start_pos, start_ang, max_iter=1000, xatol=1e-5, fatol=1e-7):
        init_params = np.zeros(3*N)
        for i in range(N):
            init_params[3*i] = start_pos[i, 0]
            init_params[3*i+1] = start_pos[i, 1]
            init_params[3*i+2] = start_ang[i]
        
        try:
            res = minimize(local_relax_objective, init_params, method='Nelder-Mead',
                           options={'maxiter': max_iter, 'xatol': xatol, 'fatol': fatol})
            
            if res.fun < 1e5:
                new_pos = np.zeros((N, 2))
                new_ang = np.zeros(N)
                for i in range(N):
                    new_pos[i, 0] = res.x[3*i]
                    new_pos[i, 1] = res.x[3*i+1]
                    new_ang[i] = res.x[3*i+2]
                
                angles_rad = np.radians(new_ang)
                for i in range(N):
                    for j in range(i+1, N):
                        h1 = (new_pos[i, 0], new_pos[i, 1], angles_rad[i])
                        h2 = (new_pos[j, 0], new_pos[j, 1], angles_rad[j])
                        if not hexagons_disjoint_fast(h1, h2):
                            return None
                
                side, phi, cx, cy = compute_min_outer_side(new_pos, new_ang)
                return (side, phi, cx, cy, new_pos, new_ang)
        except Exception:
            return None
    
    # Run polish from best solution
    if time.time() - t0 < DEADLINE * 0.85:
        polish_result = polish_from(best_positions, best_angles, max_iter=1500, xatol=1e-5, fatol=1e-7)
        if polish_result and polish_result[0] < best_side:
            best_side, best_phi, best_cx, best_cy = polish_result[0], polish_result[1], polish_result[2], polish_result[3]
            best_positions = polish_result[4]
            best_angles = polish_result[5]
    
    # Run polish from slightly perturbed versions
    if time.time() - t0 < DEADLINE * 0.97:
        for pert_i in range(3):
            if time.time() - t0 > DEADLINE * 0.97:
                break
            rng = np.random.RandomState(42 + pert_i)
            pert_pos = best_positions + rng.normal(0, 0.05, size=(N, 2))
            pert_ang = best_angles + rng.normal(0, 2.0, size=N)
            
            polish_result = polish_from(pert_pos, pert_ang, max_iter=800, xatol=1e-5, fatol=1e-7)
            if polish_result and polish_result[0] < best_side:
                best_side, best_phi, best_cx, best_cy = polish_result[0], polish_result[1], polish_result[2], polish_result[3]
                best_positions = polish_result[4]
                best_angles = polish_result[5]

    # Final Result
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle

# EVOLVE-BLOCK-END
