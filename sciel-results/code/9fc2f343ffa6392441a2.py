"""Rewritten program for hexagon packing in a hexagon using smooth optimization."""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin (evaluator limit 150s)
    
    SQRT3 = math.sqrt(3)
    
    # Helper: Get vertices of a unit hexagon centered at (cx, cy) with rotation angle
    def get_hex_vertices(cx, cy, angle_deg):
        verts = []
        for i in range(6):
            theta = math.radians(angle_deg + i * 60)
            x = cx + math.cos(theta)
            y = cy + math.sin(theta)
            verts.append((x, y))
        return verts

    # Helper: Calculate minimum distance between two convex polygons
    # If they overlap, return a negative value (penalty)
    def min_dist_poly(p1, p2):
        min_d = float('inf')
        n1 = len(p1)
        n2 = len(p2)
        
        # Check point-in-polygon for both directions
        # If p1 contains any point of p2, they overlap
        # For convex polygons, we can use SAT for exact distance or simple checks
        # For smoothness, we approximate distance. 
        # Actually, for small N and tight packings, SAT is robust.
        # Let's use a simpler smooth proxy: 
        # If we use the "separation" from SAT, it's 0 when touching.
        # We want a penalty when separation < 0 (overlap).
        # Let's compute the maximum penetration depth.
        
        axes = []
        for i in range(n1):
            p_a = p1[i]
            p_b = p1[(i + 1) % n1]
            dx = p_b[0] - p_a[0]
            dy = p_b[1] - p_a[1]
            norm = math.sqrt(dx*dx + dy*dy)
            if norm > 0:
                axes.append((-dy/norm, dx/norm))
        
        for i in range(n2):
            p_a = p2[i]
            p_b = p2[(i + 1) % n2]
            dx = p_b[0] - p_a[0]
            dy = p_b[1] - p_a[1]
            norm = math.sqrt(dx*dx + dy*dy)
            if norm > 0:
                axes.append((-dy/norm, dx/norm))
        
        max_penetration = 0.0
        for axis in axes:
            min1, max1 = float('inf'), float('-inf')
            min2, max2 = float('inf'), float('-inf')
            
            for p in p1:
                proj = p[0]*axis[0] + p[1]*axis[1]
                if proj < min1: min1 = proj
                if proj > max1: max1 = proj
                
            for p in p2:
                proj = p[0]*axis[0] + p[1]*axis[1]
                if proj < min2: min2 = proj
                if proj > max2: max2 = proj
            
            # Separation distance
            # If max1 < min2, separation is min2 - max1
            # If max2 < min1, separation is min1 - max2
            # If they overlap, separation is negative.
            sep = min(max1, max2) - max(min1, min2)
            # If sep < 0, they overlap by -sep
            if sep < max_penetration:
                max_penetration = sep
                
        # Return -max_penetration if overlapping, 0 otherwise
        # To make it smooth, we can return a softplus or just the raw negative overlap
        # Raw negative overlap is 0 at contact, <0 when overlapping.
        # We want to minimize this, so overlapping increases cost.
        # Let's return max(0, -max_penetration) * weight? 
        # No, let's just return the penetration depth if positive, 0 otherwise.
        # Penetration depth = -max_penetration if max_penetration < 0 else 0.
        if max_penetration < 0:
            return -max_penetration
        return 0.0

    # Helper: Calculate outer hexagon side length containing all hexagons
    # We optimize the outer angle as a variable in the parameter vector
    def calc_outer_side_smooth(hexagons, outer_angle_deg):
        # Outer hexagon is regular.
        # The constraint is that all points must be inside.
        # For a regular hexagon with side S and angle theta, the apothem is A = S * sqrt(3) / 2.
        # The distance from center to any edge is A.
        # The max projection of any point onto the 6 normal directions must be <= A.
        
        max_proj = 0.0
        base_angles = [30, 90, 150, 210, 270, 330]
        
        for hex_verts in hexagons:
            for vx, vy in hex_verts:
                for a in base_angles:
                    rad = math.radians(a + outer_angle_deg)
                    nx = math.cos(rad)
                    ny = math.sin(rad)
                    proj = abs(vx * nx + vy * ny)
                    if proj > max_proj:
                        max_proj = proj
        
        # S = 2 * max_proj / sqrt(3)
        return 2.0 * max_proj / SQRT3

    # Objective function
    def objective(p):
        centers = []
        angles = []
        for i in range(N):
            centers.append((p[i*3], p[i*3+1]))
            angles.append(p[i*3+2])
        
        # Center the configuration to remove translational symmetry
        cx_avg = sum(c[0] for c in centers) / N
        cy_avg = sum(c[1] for c in centers) / N
        shifted_centers = [(x - cx_avg, y - cy_avg) for x, y in centers]
        
        hexagons = []
        for i in range(N):
            cx, cy = shifted_centers[i]
            a = angles[i]
            hex_verts = get_hex_vertices(cx, cy, a)
            hexagons.append(hex_verts)
        
        # Overlap penalty
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = min_dist_poly(hexagons[i], hexagons[j])
                # Square the penalty to make it smoother and more significant when large
                overlap_penalty += d * d
        
        # Outer side length
        # We optimize the outer angle as the last parameter
        outer_angle = p[-1]
        side = calc_outer_side_smooth(hexagons, outer_angle)
        
        return side + 10.0 * overlap_penalty

    # Initial setup: Hexagonal lattice
    def axial_to_cartesian(q, r, spacing=1.0):
        x = SQRT3 * spacing * (q + r / 2.0)
        y = 1.5 * spacing * r
        return x, y

    coords = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (2,-1), (2,1), (1,1)]
    
    initial_params = []
    for q, r in coords:
        x, y = axial_to_cartesian(q, r, spacing=1.0)
        initial_params.extend([x, y, 0.0])
    initial_params.append(0.0) # Outer angle
    
    best_params = np.array(initial_params)
    best_score = objective(best_params)
    
    # Multi-restart Nelder-Mead
    # We will run several restarts with different initial configurations or perturbations
    n_restarts = 5
    time_per_restart = (deadline - time.time()) / n_restarts
    
    for r in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Perturb initial state for restarts > 0
        if r == 0:
            p0 = best_params.copy()
        else:
            p0 = best_params + np.random.normal(0, 0.1, size=len(best_params))
            # Keep outer angle reasonable
            p0[-1] = random.uniform(0, 60)
            
        # Run Nelder-Mead
        try:
            res = minimize(objective, p0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception as e:
            pass

    # Final refinement with tighter tolerance
    if time.time() < deadline - 10.0:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass

    # Extract final result
    centers = []
    angles = []
    for i in range(N):
        centers.append((best_params[i*3], best_params[i*3+1]))
        angles.append(best_params[i*3+2])
        
    cx_avg = sum(c[0] for c in centers) / N
    cy_avg = sum(c[1] for c in centers) / N
    shifted_centers = [(x - cx_avg, y - cy_avg) for x, y in centers]
    
    hexagons = []
    for i in range(N):
        cx, cy = shifted_centers[i]
        a = angles[i]
        hex_verts = get_hex_vertices(cx, cy, a)
        hexagons.append(hex_verts)
        
    # Check validity (should be valid if penalty is low)
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if not min_dist_poly(hexagons[i], hexagons[j]) == 0.0:
                # Allow tiny floating point errors
                if min_dist_poly(hexagons[i], hexagons[j]) > 1e-6:
                    valid = False
                    break
        if not valid: break
        
    if not valid:
        # Fallback to initial lattice if optimization failed badly
        inner_list = []
        for q, r in coords:
            x, y = axial_to_cartesian(q, r, spacing=1.0)
            inner_list.append([x, y, 0.0])
        cx_avg = sum(p[0] for p in inner_list) / N
        cy_avg = sum(p[1] for p in inner_list) / N
        inner_list = [[x - cx_avg, y - cy_avg, a] for x, y, a in inner_list]
        hexagons = [get_hex_vertices(p[0], p[1], p[2]) for p in inner_list]
        best_side = calc_outer_side_smooth(hexagons, 0.0)
        return (inner_list, (0.0, 0.0), best_side, 0.0)

    # Calculate final side length and angle
    outer_angle = best_params[-1]
    best_side = calc_outer_side_smooth(hexagons, outer_angle)
    
    inner_list = []
    for i in range(N):
        inner_list.append([shifted_centers[i][0], shifted_centers[i][1], angles[i]])
        
    return (inner_list, (0.0, 0.0), best_side, outer_angle)

# EVOLVE-BLOCK-END
