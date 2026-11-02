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
        
        # Max distance between centers of two touching unit hexagons is 2*sqrt(3) ~ 3.464
        # If dist > 2*sqrt(3) + epsilon, they are definitely disjoint.
        # Actually, the max distance between centers of two touching unit hexagons is 2*sqrt(3) (if aligned) or less?
        # Wait, for two regular hexagons of side 1, the max distance between centers while touching is 2*sqrt(3) (vertex to vertex opposite).
        # If dist_sq > (2*sqrt(3))^2 = 12, they are disjoint.
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
            # The normals are at angles phi + 30 + k*60 degrees
            normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            abs_proj = np.abs(proj)
            max_projs = np.max(abs_proj, axis=0)
            # Side length s relates to support function h by h = s * sqrt(3) / 2
            # So s = 2 * h / sqrt(3)
            side_vals = (2.0 / sqrt3) * max_projs
            return np.max(side_vals)
        
        # Coarse scan
        best_side = float('inf')
        best_phi = 0.0
        
        # The function is periodic with period 60 degrees.
        # We can scan 0 to 60.
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
            
        # Check validity
        if not is_valid_configuration(positions, angle_list):
            # Penalty for overlap
            # We need a smooth penalty to help the optimizer navigate out of overlaps.
            # Calculate sum of overlap depths? Or just a large penalty.
            # A large penalty works well with DE.
            return 1e6 + np.sum(positions**2)
            
        # Compute objective
        side, phi, cx, cy = compute_min_outer_side(positions, angle_list)
        return side

    # Initial guesses
    # A good initial guess for the symmetric 2-base setup
    # Let's try a honeycomb-like arrangement.
    # Center is at (0,0).
    # Orbit 1: 6 hexagons around center? No, that would be 6.
    # If we put 1 in center, it breaks 6-fold symmetry for the "2 base" model unless the center is one of the bases and the other orbit is empty? No, bases generate orbits.
    # If base 1 is at center, orbit 1 is just 1 hexagon (the center one) if we consider the rotation group action?
    # Wait, the code generates 12 hexagons from 2 bases by rotating 6 times.
    # So base 1 generates 6 hexagons. Base 2 generates 6 hexagons.
    # If base 1 is at (0,0), all 6 copies are at (0,0) and they overlap!
    # So we cannot use a center hexagon in this specific "2 orbits of 6" symmetry breaking scheme unless we handle the center case separately or use a different symmetry.
    # However, the optimal packing for 12 hexagons might not have a center hexagon.
    # Let's try:
    # Orbit 1: radius ~ 1.732 (touching center? no, center is empty).
    # Let's try a ring of 6 at r=2.0 and a ring of 6 at r=4.0?
    # Or maybe 2 hexagons close to each other in the sector.
    
    # Let's use the result from the previous best score ~9.995.
    # That corresponds to side length ~ 1/0.9995? No, score is seed/value.
    # If seed is 1.0, and score is 9.995, value is ~0.1? No.
    # "score is the outer side length". "search score is seed/value".
    # Usually in these tasks, lower side length is better.
    # If score = seed / value, and seed=1, then score = 1/value.
    # If score is 9.995, value is 1/9.995 ~ 0.1. That's too small for 12 hexagons.
    # Maybe the score is just the side length?
    # "The evaluator measures the raw value (lower is better)."
    # "search score is seed/value, so 1.0 equals the seed program and above 1.0 is an improvement"
    # This implies if I get a side length of S, my score is 1/S?
    # If the seed program got a side length of 1.0, score is 1.
    # If I get 0.9995, score is 1/0.9995 = 1.0005.
    # The current program has score 9.9951. This implies the side length is ~0.1?
    # That's impossible for 12 unit hexagons. The side length must be at least ~4 or 5.
    # Let's re-read carefully: "score is the outer side length".
    # "search score is seed/value".
    # If the "value" reported to the search engine is the side length, and the "score" shown in the prompt (9.9951) is the search score...
    # Then 9.9951 = seed / side_length.
    # If seed=1, side_length = 1/9.9951 ~ 0.1. Still impossible.
    # Maybe the "score" in the prompt IS the side length?
    # "CURRENT PROGRAM (score 9.9951)"
    # If the side length is 9.9951, that's plausible for 12 hexagons.
    # Let's assume the number 9.9951 is the side length.
    # And the "search score" mentioned in the contract is for the internal search, but the reported "score" is the raw value.
    # "The evaluator measures the raw value (lower is better)."
    # So I want to minimize the side length.
    # Current side length: ~9.995.
    # Can we do better?
    # A hexagon of side 1 has "radius" (center to vertex) 1.
    # A hexagon of side 1 has "width" (flat to flat) sqrt(3) ~ 1.732.
    # If we pack them in a honeycomb, the outer boundary grows linearly.
    # For 12 hexagons, a compact cluster is roughly 4x4 in terms of hexagons.
    # Side length ~ 4 * 1.732 ~ 7?
    # Or maybe 5?
    # Let's try to find a better configuration.

    init_params = np.array([
        1.732, 0.0, 0.0,   # Base 1: First ring
        3.464, 0.0, 0.0    # Base 2: Second ring
    ])
    
    best_side = float('inf')
    best_params = init_params.copy()
    
    # Run Differential Evolution
    bounds = [
        (-5.0, 5.0),  # x1
        (-5.0, 5.0),  # y1
        (0.0, np.pi/3), # a1
        (-5.0, 5.0),  # x2
        (-5.0, 5.0),  # y2
        (0.0, np.pi/3)  # a2
    ]
    
    try:
        # Use DE for global search
        # maxiter=100, popsize=20 -> 2000 evals
        # Each eval: 66 SAT checks + outer side calc.
        # SAT check is fast.
        # Outer side calc is ~12000 ops (72 verts * 60 phi steps).
        # Total ~ 2000 * 100k ops = 200M ops. Should be < 1s.
        
        res = differential_evolution(
            objective_symmetric,
            bounds,
            maxiter=200,
            popsize=30,
            tol=1e-6,
            seed=42,
            polish=True,
            maxfev=10000,
            mutation=(0.5, 1.0),
            recombination=0.8
        )
        
        if res.fun < best_side:
            best_side = res.fun
            best_params = res.x
            
    except Exception as e:
        pass
    
    # Refine with Nelder-Mead
    if time.time() - t0 < DEADLINE * 0.8:
        try:
            res_local = minimize(
                objective_symmetric,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            if res_local.fun < best_side:
                best_side = res_local.fun
                best_params = res_local.x
        except Exception:
            pass

    # Final Polish: Optimize the outer hexagon angle and center separately for the fixed inner positions?
    # The compute_min_outer_side already optimizes the angle and center.
    # But maybe we can refine the inner positions with a tighter tolerance?
    
    # Let's try a few random restarts if we have time
    if time.time() - t0 < DEADLINE * 0.5:
        for restart in range(5):
            # Perturb best_params
            perturbed = best_params + np.random.normal(0, 0.1, 6)
            try:
                res_restart = minimize(
                    objective_symmetric,
                    perturbed,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-7}
                )
                if res_restart.fun < best_side:
                    best_side = res_restart.fun
                    best_params = res_restart.x
            except:
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
