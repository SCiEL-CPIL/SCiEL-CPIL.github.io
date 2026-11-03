"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    sqrt3 = math.sqrt(3)

    # Helper to calculate the minimum side length S of an outer hexagon
    # centered at (0,0) with angle theta (degrees) that contains N unit hexagons
    # centered at positions (x_i, y_i) with angle 0 (aligned).
    def calc_min_side(positions, theta_deg):
        # positions: list of (x, y)
        theta_rad = math.radians(theta_deg)
        
        # Normals of the outer hexagon
        # For a hexagon with angle 0 (flat top? pointy top?), let's define consistent geometry.
        # Standard: Vertices at angles 0, 60, ...
        # Normals at 30, 90, ...
        # If we rotate the outer hexagon by theta, normals are at 30+theta, 90+theta, ...
        
        max_proj = 0.0
        for k in range(6):
            normal_angle = math.radians(30 + k * 60) + theta_rad
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            
            # Max projection of inner hexagons in this direction
            # Inner hexagon is unit side, angle 0.
            # Support function of unit hexagon in direction u is:
            # h(u) = max( |u_x|, |u_y| * 2/sqrt(3) ... )? 
            # Actually, for a unit hexagon (side 1, angle 0), the inradius is sqrt(3)/2.
            # The support function h(u) is the distance from center to the supporting line.
            # For a regular hexagon, h(u) depends on the angle of u relative to the hexagon.
            # If u is along a normal (30, 90...), h = sqrt(3)/2.
            # If u is along a vertex (0, 60...), h = 1.
            # We can calculate h(u) for a unit hexagon analytically or just use the max vertex projection.
            # Since N is small, we can just check the 6 vertices of a unit hexagon.
            
            # Unit hexagon vertices (angle 0):
            # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
            
            max_inner_proj = 0.0
            for x, y in positions:
                # We need max over all vertices of the i-th hexagon
                # v_j = (x, y) + R * v_unit_j
                # proj = (x, y) . n + R * v_unit_j . n
                # max over j of (R * v_unit_j . n) is h_unit(n)
                
                # Calculate h_unit(n)
                # n is a unit vector.
                # h_unit(n) = max_k (v_k . n)
                # We can precompute the 6 vertex vectors of a unit hexagon.
                
                pass # Placeholder

            # Let's precompute unit hexagon vertices outside this loop for efficiency
            # But for now, let's use the property that for aligned hexagons,
            # the "radius" in any direction is between sqrt(3)/2 and 1.
            # Specifically, h_unit(n) = 1 / (2 * cos(delta)) where delta is angle between n and closest normal?
            # No, simpler:
            # The support function of a regular hexagon with circumradius R=1 is:
            # h(u) = R * max(cos(alpha), cos(60-alpha)...)?
            
            # Let's just calculate it directly for accuracy.
            unit_verts = [
                (1.0, 0.0),
                (0.5, sqrt3/2),
                (-0.5, sqrt3/2),
                (-1.0, 0.0),
                (-0.5, -sqrt3/2),
                (0.5, -sqrt3/2)
            ]
            
            h_unit = 0.0
            for ux, uy in unit_verts:
                proj = ux * nx + uy * ny
                if proj > h_unit:
                    h_unit = proj
            
            # Now max over all centers
            for x, y in positions:
                proj = x * nx + y * ny + h_unit
                if proj > max_proj:
                    max_proj = proj
        
        # The outer hexagon has inradius R_in = S * sqrt(3)/2
        # We require R_in >= max_proj
        # S * sqrt(3)/2 >= max_proj
        # S >= max_proj * 2 / sqrt(3)
        
        return max_proj * 2.0 / sqrt3

    # Initial guess: Honeycomb lattice (4 rows x 3 cols)
    positions = []
    for row in range(4):
        y = row * sqrt3
        for col in range(3):
            if row % 2 == 0:
                x = (col - 1) * 2.0  # -2, 0, 2
            else:
                x = (col - 1) * 2.0 + 1.0  # -1, 1, 3
            positions.append((x, y))

    # Center the arrangement
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx_offset = (min(xs) + max(xs)) / 2.0
    cy_offset = (min(ys) + max(ys)) / 2.0
    centered_positions = [(x - cx_offset, y - cy_offset) for x, y in positions]
    
    # Convert to flat list for optimization
    init_flat = []
    for x, y in centered_positions:
        init_flat.extend([x, y])
    init_flat.append(0.0) # theta

    # Objective function for optimizer
    # params: x0, y0, ..., x11, y11, theta
    def objective(params):
        if time.time() > deadline:
            return 1e10
        
        pos = [(params[i], params[i+1]) for i in range(0, 2*N, 2)]
        theta = params[2*N]
        
        # Calculate min side length
        S = calc_min_side(pos, theta)
        
        # Penalty for overlaps
        # For aligned hexagons, min distance between centers should be >= 2
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < 2.0:
                    penalty += (2.0 - dist) ** 2
        
        return S + 100.0 * penalty

    # Optimize
    best_params = init_flat[:]
    best_score = objective(init_flat)
    
    try:
        res = minimize(objective, init_flat, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
    except Exception:
        pass
    
    # Extract results
    pos = [(best_params[i], best_params[i+1]) for i in range(0, 2*N, 2)]
    theta = best_params[2*N]
    
    # Calculate final side length
    final_side = calc_min_side(pos, theta)
    
    # Validate and adjust if necessary
    # The evaluator requires valid configuration.
    # If penalty was used, S might be slightly off.
    # We return the calculated S.
    
    inner = []
    for x, y in pos:
        inner.append([x, y, 0.0])
        
    return inner, (0.0, 0.0), final_side, theta
# EVOLVE-BLOCK-END
