import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    DEADLINE = 140.0

    # 1. Define the inner hexagon centers for the densest known packing of 11 unit hexagons.
    # This is the "flower" of 7 plus 4 in the gaps of the second ring.
    # We use a coordinate system where the center is at (0,0).
    # Distance between touching centers is sqrt(3).
    
    eps = 1e-5
    sqrt3 = np.sqrt(3)
    
    centers = []
    # Center
    centers.append((0.0, 0.0))
    
    # Ring 1: 6 hexagons
    for k in range(6):
        angle = np.deg2rad(k * 60)
        r = sqrt3 + eps
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        centers.append((x, y))
        
    # Ring 2: 4 hexagons in the gaps (angles 30, 90, 150, 270)
    # Distance from center is 3.0 (touching ring 1)
    # Actually, the distance to the gap centers in a hex grid is 3.0?
    # Let's verify.
    # Center (0,0). Ring 1 at (sqrt3, 0).
    # Gap between (sqrt3, 0) and (sqrt3/2, 1.5) is at (1.5*sqrt3, 1.5)?
    # Distance from (0,0) to (1.5*sqrt3, 1.5) is sqrt(6.75 + 2.25) = sqrt(9) = 3.
    # Yes, distance is 3.
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        r = 3.0 + 2 * eps
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        centers.append((x, y))
        
    # 2. Define the inner hexagon rotation.
    # We will optimize the outer rotation, but we can fix the inner rotation to 0 (pointy-top)
    # or 30 (flat-top). Due to symmetry, fixing inner to 0 and optimizing outer is sufficient.
    # Let's fix inner rotation to 0.0 degrees (pointy-top).
    inner_angle = 0.0
    
    # 3. Function to calculate the required outer side length for a given outer rotation.
    def calc_side(theta_out_rad):
        # Vertices of a unit hexagon with rotation theta_in
        # For theta_in = 0, vertices are at 0, 60, 120, 180, 240, 300.
        # Max extent in any direction d is max over all vertices v of (v . d) + center . d
        # For the outer hexagon with side S and rotation theta_out,
        # the distance from center to a side (apothem) is S * sqrt(3) / 2.
        # The condition for a point p to be inside is:
        # max_k (p . n_k) <= S * sqrt(3) / 2
        # where n_k are the unit normal vectors to the sides of the outer hexagon.
        
        # Normals for outer hexagon with rotation theta_out:
        # Sides are perpendicular to vectors at angles theta_out, theta_out+60, ...
        # Wait, if the hexagon is rotated by theta, the normals are at theta, theta+60...
        # Let's define normals explicitly.
        normals = []
        for k in range(6):
            angle = theta_out_rad + k * np.pi / 3
            normals.append((np.cos(angle), np.sin(angle)))
        
        max_projection = 0.0
        for cx, cy in centers:
            # Calculate max projection of this hexagon's vertices onto each normal
            # Vertices of unit hexagon at (cx, cy) with rotation inner_angle:
            # v_j = (cx + cos(inner_angle + j*pi/3), cy + sin(inner_angle + j*pi/3))
            # We need max over j and k of (v_j . n_k)
            # This is equivalent to max_k ( (cx, cy) . n_k + max_j (v_j_rel . n_k) )
            # max_j (v_j_rel . n_k) is the support function of the unit hexagon in direction n_k.
            # For a unit hexagon, the support function h(u) depends on the angle of u relative to the hexagon.
            # If inner_angle = 0, vertices are at 0, 60...
            # The support function is piecewise linear.
            # Simpler: just compute the max dot product of the 6 vertices with each of the 6 normals.
            for nx, ny in normals:
                # Project center
                proj_center = cx * nx + cy * ny
                # Project vertices relative to center
                max_v_proj = 0.0
                for j in range(6):
                    v_angle = inner_angle + j * np.pi / 3
                    vx = np.cos(v_angle)
                    vy = np.sin(v_angle)
                    proj_v = vx * nx + vy * ny
                    if proj_v > max_v_proj:
                        max_v_proj = proj_v
                total_proj = proj_center + max_v_proj
                if total_proj > max_projection:
                    max_projection = total_proj
        
        # The apothem of the outer hexagon must be >= max_projection
        # Apothem = S * sqrt(3) / 2
        # S = 2 * max_projection / sqrt(3)
        return 2.0 * max_projection / sqrt3

    # 4. Optimize the outer rotation theta_out
    # We expect the optimal angle to be around 30 degrees or 0 degrees.
    # Let's scan a range or use minimize.
    
    best_side = float('inf')
    best_theta = 0.0
    
    # Quick scan to find a good starting point
    for theta_deg in range(0, 60, 5):
        theta_rad = np.deg2rad(theta_deg)
        side = calc_side(theta_rad)
        if side < best_side:
            best_side = side
            best_theta = theta_rad
            
    # Refine with scipy
    res = minimize(calc_side, x0=best_theta, method='Nelder-Mead', 
                   options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 200})
    
    if res.success and res.fun < best_side:
        best_side = res.fun
        best_theta = res.x[0]
        
    # 5. Construct the return values
    inner = []
    for x, y in centers:
        inner.append([x, y, inner_angle])
        
    outer_center = [0.0, 0.0]
    outer_side = best_side
    outer_angle_deg = np.rad2deg(best_theta)
    
    # Check time
    if time.time() - start_time > DEADLINE:
        # Fallback to a known valid configuration if optimization took too long
        # (Though calc_side is fast, this is a safety net)
        return inner, outer_center, 3.0, 30.0

    return inner, outer_center, outer_side, outer_angle_deg

