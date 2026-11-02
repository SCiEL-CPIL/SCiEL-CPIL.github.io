import math
import time
import random
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 12
DEADLINE = 145.0  # Safety margin before 150s wall-clock kill

def construct():
    t0 = time.time()
    
    # Precompute unit hexagon vertices (angle 0)
    angles = np.radians(np.arange(0, 360, 60))
    unit_hex_verts = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Helper: Get vertices of a single hexagon
    def get_hex_verts(cx, cy, angle_rad):
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        ux = unit_hex_verts[:, 0]
        uy = unit_hex_verts[:, 1]
        rx = ux * c - uy * s
        ry = ux * s + uy * c
        return np.column_stack((rx + cx, ry + cy))

    # Helper: Get outward normals for a CCW polygon
    def get_normals(verts):
        edges = np.roll(verts, -1, axis=0) - verts
        normals = np.column_stack([edges[:, 1], -edges[:, 0]])
        norms = np.linalg.norm(normals, axis=1)
        normals = normals / (norms[:, np.newaxis] + 1e-15)
        return normals

    # Fast SAT check for two unit hexagons - returns max overlap depth
    def hexagons_overlap_depth(h1, h2):
        """
        h1, h2: (x, y, angle_rad)
        Returns max overlap depth (positive if overlapping, 0 if disjoint).
        """
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        
        # Max distance between centers of two touching unit hexagons is 2*sqrt(3) ~ 3.464
        if dist_sq > 12.0:
            return 0.0
            
        v1 = get_hex_verts(h1[0], h1[1], h1[2])
        v2 = get_hex_verts(h2[0], h2[1], h2[2])
        
        n1 = get_normals(v1)
        n2 = get_normals(v2)
        normals = np.vstack([n1, n2])
        
        p1 = v1 @ normals.T
        p2 = v2 @ normals.T
        
        min1 = p1.min(axis=0)
        max1 = p1.max(axis=0)
        min2 = p2.min(axis=0)
        max2 = p2.max(axis=0)
        
        # Overlap depth for each normal
        # If intervals [min1, max1] and [min2, max2] overlap, the depth is min(max1, max2) - max(min1, min2)
        # We want the maximum of this over all normals. If it is > 0, they overlap.
        overlaps = np.minimum(max1, max2) - np.maximum(min1, min2)
        
        max_overlap = np.max(overlaps)
        if max_overlap < 0:
            return 0.0
        return max_overlap

    def compute_min_outer_side(positions, angles_deg):
        """
        Compute the minimum side length of a regular hexagon containing all inner hexagons.
        Returns (side_length, optimal_outer_angle_deg, center_x, center_y)
        """
        angles_rad = np.radians(angles_deg)
        
        # Collect all vertices
        all_verts = []
        for i in range(N):
            v = get_hex_verts(positions[i, 0], positions[i, 1], angles_rad[i])
            all_verts.append(v)
        verts_flat = np.vstack(all_verts)
        
        # Center
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        sqrt3 = math.sqrt(3)
        best_side = float('inf')
        best_phi = 0.0
        
        def eval_phi(phi_deg):
            phi = np.radians(phi_deg)
            # 6 normal directions for the outer hexagon
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
        """Check if all hexagons are pairwise disjoint."""
        angles_rad = np.radians(angles_deg)
        for i in range(N):
            for j in range(i+1, N):
                h1 = (positions[i, 0], positions[i, 1], angles_rad[i])
                h2 = (positions[j, 0], positions[j, 1], angles_rad[j])
                if hexagons_overlap_depth(h1, h2) > 1e-9:
                    return False
        return True

    # Objective function for optimization
    # We use 6-fold symmetry: define 2 base hexagons, generate 12 by rotating.
    def objective_symmetric(params):
        if time.time() - t0 > DEADLINE:
            return 1e10
            
        x1, y1, a1, x2, y2, a2 = params
        
        # Generate the 12 hexagons
        positions = np.zeros((N, 2))
        angle_list = np.zeros(N)
        
        for k in range(6):
            rot_angle = np.radians(60 * k)
            c = math.cos(rot_angle)
            s = math.sin(rot_angle)
            
            # Rotate base1
            x1_rot = x1 * c - y1 * s
            y1_rot = x1 * s + y1 * c
            a1_rot = a1 + rot_angle
            
            # Rotate base2
            x2_rot = x2 * c - y2 * s
            y2_rot = x2 * s + y2 * c
            a2_rot = a2 + rot_angle
            
            positions[2*k, 0] = x1_rot
            positions[2*k, 1] = y1_rot
            angle_list[2*k] = np.degrees(a1_rot)
            
            positions[2*k+1, 0] = x2_rot
            positions[2*k+1, 1] = y2_rot
            angle_list[2*k+1] = np.degrees(a2_rot)
            
        # Calculate penalty for overlaps
        angles_rad = np.radians(angle_list)
        total_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                h1 = (positions[i, 0], positions[i, 1], angles_rad[i])
                h2 = (positions[j, 0], positions[j, 1], angles_rad[j])
                depth = hexagons_overlap_depth(h1, h2)
                if depth > 0:
                    total_penalty += depth * depth
        
        # Compute objective
        side, phi, cx, cy = compute_min_outer_side(positions, angle_list)
        
        # Combined objective: Side + Penalty
        # The penalty should be large enough to prevent overlaps but small enough to allow movement.
        # If side is ~5, and overlap is 0.1, penalty is 0.01.
        # We want to minimize side primarily.
        # However, if we start with overlaps, we need to push them out.
        # A large multiplier helps.
        return side + 10.0 * total_penalty

    # Initial guesses
    init_params = np.array([
        2.0, 0.0, 0.0,   # Base 1
        4.0, 0.0, 0.0    # Base 2
    ])
    
    best_side = float('inf')
    best_params = init_params.copy()
    
    # Run Differential Evolution
    bounds = [
        (-5.0, 5.0),  # x1
        (-5.0, 5.0),  # y1
        (0.0, np.pi/3), # a1 (0 to 60 deg)
        (-5.0, 5.0),  # x2
        (-5.0, 5.0),  # y2
        (0.0, np.pi/3)  # a2
    ]
    
    try:
        res = differential_evolution(
            objective_symmetric,
            bounds,
            maxiter=150,
            popsize=25,
            tol=1e-5,
            seed=42,
            polish=True,
            maxfev=8000
        )
        
        if res.fun < best_side:
            best_side = res.fun
            best_params = res.x
            
    except Exception as e:
        pass

    # If DE didn't find a good solution or we have time left, run SA on the full space
    # Or run SA on the symmetric space to refine.
    if time.time() - t0 < DEADLINE * 0.7:
        current_params = best_params.copy()
        current_val = objective_symmetric(current_params)
        
        T = 0.1
        T_min = 0.001
        cooling = 0.995
        step_size = 0.1
        
        for step in range(5000):
            if time.time() - t0 > DEADLINE * 0.8:
                break
                
            idx = random.randint(0, 5)
            new_params = current_params.copy()
            
            if idx < 2: # Position
                new_params[idx] += random.gauss(0, step_size)
            else: # Angle
                new_params[idx] += random.gauss(0, step_size * 0.5)
                
            new_val = objective_symmetric(new_params)
            
            if new_val < current_val:
                current_params = new_params
                current_val = new_val
                if current_val < best_side:
                    best_side = current_val
                    best_params = new_params.copy()
            else:
                delta = new_val - current_val
                if delta < 0: delta = 0
                if random.random() < math.exp(-delta / T):
                    current_params = new_params
                    current_val = new_val
                    if current_val < best_side:
                        best_side = current_val
                        best_params = new_params.copy()
                        
            T *= cooling
            if T < T_min: T = T_min

    # Final Polish with Nelder-Mead on the symmetric parameters
    if time.time() - t0 < DEADLINE * 0.95:
        try:
            res_local = minimize(
                objective_symmetric,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-7}
            )
            
            if res_local.fun < best_side:
                best_side = res_local.fun
                best_params = res_local.x
        except Exception:
            pass

    # Construct the final answer from best_params
    x1, y1, a1, x2, y2, a2 = best_params
    
    inner = []
    for k in range(6):
        rot_angle = np.radians(60 * k)
        c = math.cos(rot_angle)
        s = math.sin(rot_angle)
        
        # Rotate base1
        x1_rot = x1 * c - y1 * s
        y1_rot = x1 * s + y1 * c
        a1_rot = a1 + rot_angle
        
        # Rotate base2
        x2_rot = x2 * c - y2 * s
        y2_rot = x2 * s + y2 * c
        a2_rot = a2 + rot_angle
        
        inner.append([float(x1_rot), float(y1_rot), float(np.degrees(a1_rot))])
        inner.append([float(x2_rot), float(y2_rot), float(np.degrees(a2_rot))])
        
    # Compute the actual minimum outer side length and center
    positions = np.zeros((N, 2))
    angle_list = np.zeros(N)
    for i in range(N):
        positions[i, 0] = inner[i][0]
        positions[i, 1] = inner[i][1]
        angle_list[i] = inner[i][2]
        
    best_side, best_phi, best_cx, best_cy = compute_min_outer_side(positions, angle_list)
    
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle

# EVOLVE-BLOCK-END
