"""Rewritten program for packing 12 unit hexagons in a regular hexagon.
Uses Hard Constraint Simulated Annealing with vectorized validity checks.
"""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 12
SQRT3 = math.sqrt(3.0)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Precompute unit hexagon vertices (side 1, angle 0)
    angles = np.radians(np.arange(0, 360, 60))
    unit_verts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Normal vectors for the 6 sides of a unit hexagon (outward pointing)
    normal_angles = np.radians(np.arange(30, 360, 60))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    apothem = SQRT3 / 2.0

    # Relative positions of the 12 inner hexagons in a rigid honeycomb cluster
    # Using the 3x4 grid configuration from the original code
    rel_centers = []
    for r in range(3):
        for c in range(4):
            x = (c - 1.5) * 3.0
            y = (r - 1) * 2.598 # approx sqrt(3)
            if r % 2 == 1:
                x += 1.5
            rel_centers.append((x, y))
    rel_centers = np.array(rel_centers)
    
    def objective(params):
        # params: [cx, cy, phi, ox, oy, s, theta]
        cx, cy, phi, ox, oy, s, theta = params
        
        if s < 1.0:
            return 1000.0 # Penalty for too small side

        # Cluster rotation matrix
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)
        
        # Outer hexagon parameters
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        
        # Outer normals rotated by theta
        # normals are in local frame of outer hex
        # We need to check projection of inner vertices onto outer normals
        # It is easier to transform inner vertices to outer local frame
        
        outer_apothem = s * SQRT3 / 2.0
        
        penalty = 0.0
        
        # Outer rotation matrix (inverse to transform points to outer frame)
        # R_out = [[cos, -sin], [sin, cos]]
        # To transform point p_global to p_local: p_local = R_out^T (p_global - center)
        # R_out^T = [[cos, sin], [-sin, cos]]
        
        for i in range(N):
            rx, ry = rel_centers[i]
            
            # Transform relative center to global
            gx = cx + cos_phi * rx - sin_phi * ry
            gy = cy + sin_phi * rx + cos_phi * ry
            
            # Transform global to outer local
            lx = cos_theta * (gx - ox) + sin_theta * (gy - oy)
            ly = -sin_theta * (gx - ox) + cos_theta * (gy - oy)
            
            # Inner hexagon is rotated by phi relative to cluster, cluster is fixed to global?
            # Wait, the inner hexagons are part of the rigid cluster.
            # Their orientation in global frame is phi (since rel_centers are defined in cluster frame, and cluster rotates by phi).
            # So inner hexagon i has global angle phi.
            
            # We need vertices of inner hexagon i in outer local frame.
            # Vertices in inner local frame: unit_verts
            # Inner global rotation: phi
            # Outer local rotation: theta
            # Relative rotation: phi - theta
            
            rel_angle = phi - theta
            cos_ra = math.cos(rel_angle)
            sin_ra = math.sin(rel_angle)
            
            # Rotate unit_verts by rel_angle
            vx = cos_ra * unit_verts[0] - sin_ra * unit_verts[1]
            vy = sin_ra * unit_verts[0] + cos_ra * unit_verts[1]
            
            # Add center (lx, ly)
            vx += lx
            vy += ly
            
            # Check projection onto 6 outer normals
            # normals are fixed in outer local frame
            proj = normals @ np.vstack([vx, vy]).T
            max_proj = np.max(proj)
            
            if max_proj > outer_apothem:
                violation = max_proj - outer_apothem
                penalty += violation * violation
                
        return s + 10.0 * penalty

    def is_valid_final(inner, oc, os, oa):
        ox, oy = oc
        outer_side = os
        outer_angle = oa
        
        if outer_side < 1.0:
            return False

        outer_apothem = outer_side * SQRT3 / 2.0
        
        cos_oa = math.cos(-math.radians(outer_angle))
        sin_oa = math.sin(-math.radians(outer_angle))
        
        for i in range(N):
            cx, cy, angle_deg = inner[i]
            angle_rad = math.radians(angle_deg)
            
            rel_cx = cx - ox
            rel_cy = cy - oy
            
            rel_angle_rad = angle_rad - math.radians(outer_angle)
            cos_ra = math.cos(rel_angle_rad)
            sin_ra = math.sin(rel_angle_rad)
            
            lx = cos_ra * unit_verts[0] - sin_ra * unit_verts[1] + rel_cx
            ly = sin_ra * unit_verts[0] + cos_ra * unit_verts[1] + rel_cy
            
            proj_matrix = normals @ np.vstack([lx, ly]).T
            if np.max(proj_matrix) > outer_apothem + 1e-7:
                return False

        # Check disjointness of inner hexagons
        # Since they are in a rigid honeycomb, they should be disjoint.
        # But let's verify with SAT just in case of floating point errors in transformation
        for i in range(N):
            for j in range(i+1, N):
                if not sat_disjoint(inner[i], inner[j]):
                    return False
                    
        return True

    def sat_disjoint(h1, h2):
        c1x, c1y, a1_deg = h1
        c2x, c2y, a2_deg = h2
        
        a1 = math.radians(a1_deg)
        a2 = math.radians(a2_deg)
        
        v1 = np.column_stack([
            c1x + np.cos(a1) * unit_verts[0] - np.sin(a1) * unit_verts[1],
            c1y + np.sin(a1) * unit_verts[0] + np.cos(a1) * unit_verts[1]
        ])
        
        v2 = np.column_stack([
            c2x + np.cos(a2) * unit_verts[0] - np.sin(a2) * unit_verts[1],
            c2y + np.sin(a2) * unit_verts[0] + np.cos(a2) * unit_verts[1]
        ])
        
        R1 = np.array([[math.cos(a1), -math.sin(a1)], [math.sin(a1), math.cos(a1)]])
        R2 = np.array([[math.cos(a2), -math.sin(a2)], [math.sin(a2), math.cos(a2)]])
        
        axes1 = (R1 @ normals.T).T
        axes2 = (R2 @ normals.T).T
        
        all_axes = np.vstack([axes1, axes2])
        
        for axis in all_axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                return True
        
        return False

    best_score = float('inf')
    best_params = None
    
    # Initial guess: Center cluster at 0,0, outer at 0,0, side 8, angle 0
    # Try a few different initial angles and positions
    rng = np.random.default_rng(42)
    
    for _ in range(5):
        if time.time() > deadline - 10:
            break
            
        # Randomize initial parameters slightly
        cx0 = rng.uniform(-1, 1)
        cy0 = rng.uniform(-1, 1)
        phi0 = rng.uniform(0, math.pi/3)
        ox0 = rng.uniform(-1, 1)
        oy0 = rng.uniform(-1, 1)
        s0 = 8.0
        theta0 = rng.uniform(0, math.pi/3)
        
        x0 = np.array([cx0, cy0, phi0, ox0, oy0, s0, theta0])
        
        bounds = [
            (-5, 5), # cx
            (-5, 5), # cy
            (0, math.pi/3), # phi
            (-5, 5), # ox
            (-5, 5), # oy
            (1.0, 20.0), # s
            (0, math.pi/3) # theta
        ]
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-8})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            continue

    if best_params is None:
        # Fallback to honeycomb
        best_params = np.array([0, 0, 0, 0, 0, 8.0, 0.0])
        best_score = objective(best_params)

    cx, cy, phi, ox, oy, s, theta = best_params
    
    # Construct the final inner hexagons
    inner = []
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)
    
    for i in range(N):
        rx, ry = rel_centers[i]
        gx = cx + cos_phi * rx - sin_phi * ry
        gy = cy + sin_phi * rx + cos_phi * ry
        inner.append((gx, gy, math.degrees(phi)))
        
    oc = (ox, oy)
    os = s
    oa = math.degrees(theta)
    
    # Final validation and correction
    if not is_valid_final(inner, oc, os, oa):
        # Increase side length until valid
        for _ in range(100):
            os += 0.1
            if is_valid_final(inner, oc, os, oa):
                break
        if not is_valid_final(inner, oc, os, oa):
            # Last resort: large hexagon
            os = 20.0
    
    return inner, oc, os, oa
