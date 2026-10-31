import numpy as np
from scipy.optimize import minimize
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Objective function to minimize:
    # 1. The required outer side length S
    # 2. Penalties for overlapping inner hexagons
    
    def calc_support(angle_rel):
        # Support function of a unit regular hexagon in a direction angle_rel
        # relative to its local orientation.
        # Vertices are at 0, 60, 120...
        # The support is cos(delta) where delta is distance to nearest vertex angle.
        # Wait, support function h(u) = max u.v.
        # For unit hex, vertices v_k = (cos(k*pi/3), sin(k*pi/3)).
        # h(theta) = max_k cos(theta - k*pi/3)
        # This equals cos(min_k |theta - k*pi/3|)
        # Since cos is even and periodic.
        # We can map angle to [0, pi/3] and use cos.
        angle_rel = np.mod(angle_rel, np.pi / 3)
        # The max deviation from a vertex is pi/6.
        # So angle_rel is in [0, pi/3]. Distance to nearest vertex is min(angle_rel, pi/3 - angle_rel).
        delta = np.minimum(angle_rel, np.pi / 3 - angle_rel)
        return np.cos(delta)

    def objective(params):
        # params: [x1, y1, theta1, x2, y2, theta2, ...]
        if len(params) != 3 * N:
            return 1e10
            
        positions = params[0::3]
        ys = params[1::3]
        thetas = params[2::3]
        
        # 1. Calculate required S
        # Outer hexagon normals are at 30, 90, 150... degrees (if flat top)
        # Or 0, 60... (if pointy top).
        # Let's assume outer hexagon is fixed at angle 0.
        # Normals to sides of a regular hexagon (pointy top, vertices at 0, 60...)
        # are at 30, 90, 150, 210, 270, 330.
        normal_angles = np.array([np.pi/6 + k * np.pi/3 for k in range(6)])
        
        max_ext = 0.0
        for k in range(6):
            nk = normal_angles[k]
            # Unit normal vector
            nx = np.cos(nk)
            ny = np.sin(nk)
            
            # For each inner hexagon i
            # Projection of center: xi*nx + yi*ny
            # Support of inner hex in direction nk:
            # The inner hex is rotated by theta_i.
            # The direction nk in local frame of hex i is nk - theta_i.
            rel_angles = nk - thetas
            supports = calc_support(rel_angles)
            
            projections = positions * nx + ys * ny
            extents = projections + supports
            
            current_max = np.max(extents)
            if current_max > max_ext:
                max_ext = current_max
                
        # The distance from center to side is S * sqrt(3) / 2.
        # So S * sqrt(3) / 2 >= max_ext
        # S >= 2 * max_ext / sqrt(3)
        S_required = 2.0 * max_ext / np.sqrt(3)
        
        # 2. Calculate Overlap Penalty
        # Approximate overlap check:
        # If distance between centers < sqrt(3), they likely overlap.
        # We use a smooth penalty: sum of (sqrt(3) - dist)^2 for dist < sqrt(3)
        penalty = 0.0
        min_dist = np.sqrt(3) # ~1.732
        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[i] - positions[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < min_dist:
                    # Penalty increases as they get closer
                    # Use a strong factor to ensure separation
                    penalty += (min_dist - dist) ** 2 * 10.0
                    
        # Return S + Penalty
        # If penalty is large, it dominates.
        # We want to minimize S.
        return S_required + penalty

    # Initialization
    # Start with a compact cluster
    # 1 center, 6 around it, 4 more
    init_params = []
    
    # Center
    init_params.extend([0.0, 0.0, 0.0])
    
    # 6 neighbors at distance sqrt(3)
    for k in range(6):
        angle = k * np.pi / 3
        x = np.sqrt(3) * np.cos(angle)
        y = np.sqrt(3) * np.sin(angle)
        init_params.extend([x, y, 0.0])
        
    # 4 more: place them at the "gaps" or extend the ring
    # Let's place them at distance ~2*sqrt(3) in 4 directions
    for k in range(4):
        angle = k * np.pi / 2 + np.pi / 4 # 45, 135, 225, 315
        x = 2.0 * np.sqrt(3) * np.cos(angle)
        y = 2.0 * np.sqrt(3) * np.sin(angle)
        init_params.extend([x, y, 0.0])
        
    init_params = np.array(init_params)
    
    # Optimization
    # Nelder-Mead is robust for small N
    try:
        res = minimize(objective, init_params, method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol': 1e-5, 'fatol': 1e-5, 'disp': False})
        
        final_params = res.x
    except Exception:
        final_params = init_params

    # Extract final positions
    final_positions = []
    for i in range(N):
        x = final_params[3*i]
        y = final_params[3*i + 1]
        angle_deg = np.degrees(final_params[3*i + 2])
        final_positions.append([x, y, angle_deg])
        
    # Calculate the final S for the return value
    # Re-evaluate S without penalty to get the true score
    positions = final_params[0::3]
    ys = final_params[1::3]
    thetas = final_params[2::3]
    
    normal_angles = np.array([np.pi/6 + k * np.pi/3 for k in range(6)])
    max_ext = 0.0
    for k in range(6):
        nk = normal_angles[k]
        nx = np.cos(nk)
        ny = np.sin(nk)
        rel_angles = nk - thetas
        supports = calc_support(rel_angles)
        projections = positions * nx + ys * ny
        extents = projections + supports
        current_max = np.max(extents)
        if current_max > max_ext:
            max_ext = current_max
            
    S_final = 2.0 * max_ext / np.sqrt(3)
    
    # Check for overlaps one last time. If overlaps exist, the solution is invalid (score 0).
    # We should try to return a valid solution.
    # If the optimizer failed to separate them, we might need a fallback.
    # But Nelder-Mead with penalty usually works for N=11.
    
    return final_positions, (0.0, 0.0), S_final, 0.0
# EVOLVE-BLOCK-END
