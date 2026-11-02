import math
import time
import random
import numpy as np
from scipy.optimize import differential_evolution, minimize, minimize_scalar

N = 12
DEADLINE = 135.0  # Safety margin before 150s wall-clock kill

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

    # Fast SAT check for two unit hexagons
    def hexagons_disjoint_fast(h1, h2):
        """
        h1, h2: (x, y, angle_rad)
        Returns True if disjoint.
        """
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        
        # If dist > 2*sqrt(3) + epsilon, they are definitely disjoint.
        if dist_sq > 12.0:
            return True
            
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
        
        overlaps = np.minimum(max1, max2) - np.maximum(min1, min2)
        
        return np.min(overlaps) <= 1e-9

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
        
        def eval_phi(phi_deg):
            phi = np.radians(phi_deg)
            # 6 normal directions for the outer hexagon
            normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            abs_proj = np.abs(proj)
            max_projs = np.max(abs_proj, axis=0)
            # Side length s relates to support function h by h = s * sqrt(3) / 2
            side_vals = (2.0 / sqrt3) * max_projs
            return np.max(side_vals)
        
        # Coarse scan
        best_side = float('inf')
        best_phi = 0.0
        
        for phi_deg in np.arange(0, 60, 0.5):
            s = eval_phi(phi_deg)
            if s < best_side:
                best_side = s
                best_phi = phi_deg
                
        # Fine scan around best
        for phi_deg in np.arange(max(0, best_phi - 0.5), min(60, best_phi + 0.5), 0.01):
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
                if not hexagons_disjoint_fast(h1, h2):
                    return False
        return True

    # Objective function for optimization
    def objective(params):
        if time.time() - t0 > DEADLINE:
            return 1e10
            
        positions = params[:2*N].reshape(N, 2)
        angles_deg = params[2*N:]
        
        # Check validity
        if not is_valid_configuration(positions, angles_deg):
            # Penalty: Sum of overlap depths is hard to compute fast, so use a large constant
            # plus a term to encourage separation
            return 1e6 + np.sum(np.abs(positions))
            
        # Compute objective
        side, phi, cx, cy = compute_min_outer_side(positions, angles_deg)
        return side

    # Initial Guess: Honeycomb arrangement
    # 1 center + 6 ring 1 + 5 ring 2 (partial)
    # Actually, let's use a 2x3 grid of hexagons which is valid and compact.
    # Hex grid coordinates to Cartesian:
    # x = 1.5 * col + 0.75 * (row % 2) * 0? No.
    # Flat-top hexagons:
    # dx = 1.5 * s, dy = sqrt(3) * s
    # Let's just place them in a line or a small cluster.
    # A safe valid start: 12 hexagons in a 4x3 grid?
    # Or simply 12 hexagons in a straight line? (Bad for side length, but valid).
    # Let's try a 2x6 grid.
    
    init_positions = np.zeros((N, 2))
    init_angles = np.zeros(N)
    
    sqrt3 = math.sqrt(3)
    # Flat-top hexagons: width = 2, height = sqrt(3)
    # Horizontal spacing: 1.5
    # Vertical spacing: sqrt(3)
    # Offset for odd columns: sqrt(3)/2
    
    idx = 0
    for row in range(2):
        for col in range(6):
            x = col * 1.5
            y = row * sqrt3
            if col % 2 == 1:
                y += sqrt3 / 2
            init_positions[idx, 0] = x
            init_positions[idx, 1] = y
            init_angles[idx] = 0
            idx += 1
            
    # Center the configuration
    center_init = np.mean(init_positions, axis=0)
    init_positions -= center_init
    
    init_params = np.concatenate([init_positions.flatten(), init_angles])
    
    best_side = float('inf')
    best_params = init_params.copy()
    
    # Check if initial is valid and get its side length
    if is_valid_configuration(init_positions, init_angles):
        s, _, _, _ = compute_min_outer_side(init_positions, init_angles)
        best_side = s
    
    # Optimize
    try:
        # Nelder-Mead is robust for this size
        res = minimize(
            objective,
            best_params,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
        )
        
        if res.fun < best_side:
            # Verify validity of the result
            res_positions = res.x[:2*N].reshape(N, 2)
            res_angles = res.x[2*N:]
            if is_valid_configuration(res_positions, res_angles):
                best_side = res.fun
                best_params = res.x
    except Exception:
        pass
    
    # If we have time, try a few random restarts from the best
    if time.time() - t0 < DEADLINE * 0.5:
        for _ in range(3):
            perturbed = best_params + np.random.normal(0, 0.1, best_params.size)
            try:
                res_restart = minimize(
                    objective,
                    perturbed,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-7}
                )
                if res_restart.fun < best_side:
                    res_positions = res_restart.x[:2*N].reshape(N, 2)
                    res_angles = res_restart.x[2*N:]
                    if is_valid_configuration(res_positions, res_angles):
                        best_side = res_restart.fun
                        best_params = res_restart.x
            except:
                pass

    # Extract final positions and angles
    final_positions = best_params[:2*N].reshape(N, 2)
    final_angles = best_params[2*N:]
    
    # Final validity check
    if not is_valid_configuration(final_positions, final_angles):
        # Fallback to initial valid configuration
        final_positions = init_positions
        final_angles = init_angles
    
    inner = []
    for i in range(N):
        inner.append([float(final_positions[i, 0]), float(final_positions[i, 1]), float(final_angles[i])])
        
    # Compute the actual minimum outer side length and center
    best_side, best_phi, best_cx, best_cy = compute_min_outer_side(final_positions, final_angles)
    
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle

# EVOLVE-BLOCK-END
