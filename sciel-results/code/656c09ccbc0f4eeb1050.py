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
    # centered at (cx, cy) with angle theta (degrees) that contains N unit hexagons
    # centered at positions (x_i, y_i) with angle 0 (aligned).
    def calc_min_side(positions, theta_deg, cx=0.0, cy=0.0):
        theta_rad = math.radians(theta_deg)
        
        max_proj = 0.0
        for k in range(6):
            normal_angle = math.radians(30 + k * 60) + theta_rad
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            
            # Support function of unit hexagon in direction (nx, ny)
            # Unit hexagon vertices (angle 0):
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
            
            # Max over all centers
            for x, y in positions:
                proj = (x - cx) * nx + (y - cy) * ny + h_unit
                if proj > max_proj:
                    max_proj = proj
        
        return max_proj * 2.0 / sqrt3

    # Generate the 12-hexagon cluster: 19-hex cluster minus 6 corners minus 1 side hex
    # 19-hex cluster centers (axial to cartesian):
    # q, r -> x = q + r/2, y = r * sqrt(3)/2
    # Side 3 cluster: q, r in range -2 to 2 such that |q| <= 2, |r| <= 2, |q+r| <= 2
    
    centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q) <= 2 and abs(r) <= 2 and abs(q+r) <= 2:
                x = q + r/2.0
                y = r * sqrt3/2.0
                centers.append((x, y))
                
    # centers has 19 points.
    # Identify corners: distance from origin > 3.5?
    # Corner centers are at distance 4.
    # Side centers (non-corner) are at distance sqrt(12) ~ 3.46.
    
    corner_centers = []
    side_centers = []
    inner_centers = []
    
    for x, y in centers:
        dist = math.sqrt(x*x + y*y)
        if dist > 3.8: # Corners at 4.0
            corner_centers.append((x, y))
        elif dist > 3.0: # Side hexes at sqrt(12) ~ 3.46
            side_centers.append((x, y))
        else:
            inner_centers.append((x, y))
            
    # Remove all 6 corners
    # Remove 1 side center (e.g., the one at (3, sqrt(3)) -> x=3, y=sqrt(3))
    # Let's pick the first side center to remove
    removed_side = side_centers[0] if side_centers else None
    
    final_centers = inner_centers + [c for c in side_centers if c != removed_side]
    
    # We should have 19 - 6 - 1 = 12 centers.
    if len(final_centers) != 12:
        # Fallback to honeycomb 4x3
        positions = []
        for row in range(4):
            y = row * sqrt3
            for col in range(3):
                if row % 2 == 0:
                    x = (col - 1) * 2.0
                else:
                    x = (col - 1) * 2.0 + 1.0
                positions.append((x, y))
        final_centers = positions

    # Center the arrangement
    xs = [p[0] for p in final_centers]
    ys = [p[1] for p in final_centers]
    cx_offset = (min(xs) + max(xs)) / 2.0
    cy_offset = (min(ys) + max(ys)) / 2.0
    centered_positions = [(x - cx_offset, y - cy_offset) for x, y in final_centers]
    
    # Convert to flat list for optimization
    init_flat = []
    for x, y in centered_positions:
        init_flat.extend([x, y])
    init_flat.append(0.0) # theta

    # Objective function
    def objective(params):
        if time.time() > deadline:
            return 1e10
        
        pos = [(params[i], params[i+1]) for i in range(0, 2*N, 2)]
        theta = params[2*N]
        
        # Optimize outer center and rotation
        cx = 0.0
        cy = 0.0
        
        S = calc_min_side(pos, theta, cx, cy)
        
        # Penalty for overlaps
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < 2.0 - 1e-9:
                    penalty += (2.0 - dist) ** 2
        
        return S + 100.0 * penalty

    best_params = init_flat[:]
    best_score = objective(init_flat)
    
    try:
        res = minimize(objective, init_flat, method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
    except Exception:
        pass
    
    # Extract results
    pos = [(best_params[i], best_params[i+1]) for i in range(0, 2*N, 2)]
    theta = best_params[2*N]
    
    # Optimize outer center to minimize side length
    def outer_side_only(cx, cy):
        return calc_min_side(pos, theta, cx, cy)
    
    # Initial guess for center is centroid of inner hexagons
    cx0 = sum(x for x, y in pos) / N
    cy0 = sum(y for x, y in pos) / N
    
    try:
        center_res = minimize(
            lambda c: outer_side_only(c[0], c[1]),
            [cx0, cy0],
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        best_cx, best_cy = center_res.x
    except Exception:
        best_cx, best_cy = cx0, cy0
    
    final_side = calc_min_side(pos, theta, best_cx, best_cy)
    
    inner = []
    for x, y in pos:
        inner.append([x, y, 0.0])
        
    return inner, (best_cx, best_cy), final_side, theta
# EVOLVE-BLOCK-END
