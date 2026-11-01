"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time
import random

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # 1. Define rigid cluster of 12 unit hexagons
    # Honeycomb lattice:
    # Row 0 (y=0): x = 0, sqrt(3), 2sqrt(3), 3sqrt(3)
    # Row 1 (y=1.5): x = sqrt(3)/2, 3sqrt(3)/2, 5sqrt(3)/2, 7sqrt(3)/2
    # Row 2 (y=3): x = 0, sqrt(3), 2sqrt(3), 3sqrt(3)
    
    sqrt3 = np.sqrt(3)
    raw_positions = []
    # Row 0
    for i in range(4):
        raw_positions.append([i * sqrt3, 0.0])
    # Row 1
    for i in range(4):
        raw_positions.append([(i + 0.5) * sqrt3, 1.5])
    # Row 2
    for i in range(4):
        raw_positions.append([i * sqrt3, 3.0])
        
    raw_positions = np.array(raw_positions)
    
    # Center the cluster at (0,0)
    center = np.mean(raw_positions, axis=0)
    centered_positions = raw_positions - center
    
    # Get vertices of all 12 hexagons in this rigid configuration
    # A unit hexagon at (cx, cy) with angle 0 has vertices at:
    # (cx + cos(k*pi/3), cy + sin(k*pi/3)) for k=0..5
    # Note: Standard hexagon orientation (flat top/bottom or pointy top?)
    # The task allows rotation. The "angle" in the output is the rotation of the hexagon.
    # For the rigid cluster, we assume all inner hexagons are aligned with the lattice (angle 0 relative to cluster).
    # So we generate vertices for each hexagon at angle 0.
    
    hex_angles = np.radians(np.arange(0, 360, 60))
    base_vertices = np.column_stack([np.cos(hex_angles), np.sin(hex_angles)])
    
    all_vertices = []
    for cx, cy in centered_positions:
        verts = base_vertices + np.array([cx, cy])
        all_vertices.append(verts)
        
    all_vertices = np.vstack(all_vertices) # Shape (72, 2)
    
    # Pre-compute normals for outer hexagon
    # Outer hexagon with side S, rotation theta_o.
    # Normals are at theta_o + 30 + k*60 degrees.
    # Inequality: n . p <= S * sqrt(3) / 2
    
    def get_min_side(theta_c_deg, theta_o_deg):
        theta_c = np.radians(theta_c_deg)
        theta_o = np.radians(theta_o_deg)
        
        # Rotate inner vertices by theta_c
        R_c = np.array([
            [np.cos(theta_c), -np.sin(theta_c)],
            [np.sin(theta_c), np.cos(theta_c)]
        ])
        rotated_verts = all_vertices @ R_c.T
        
        # Calculate max projection onto 6 normals of outer hexagon
        normal_angles = theta_o + np.radians(30 + np.arange(0, 360, 60))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        # Project rotated_verts onto each normal
        # proj shape: (72, 6)
        proj = rotated_verts @ normals.T
        
        # Max projection for each normal
        max_projs = np.max(proj, axis=0) # shape (6,)
        
        # Required S for each normal: S >= max_proj / (sqrt(3)/2)
        # S = 2/sqrt(3) * max(max_projs)
        max_proj_val = np.max(max_projs)
        S = 2.0 / sqrt3 * max_proj_val
        
        return S

    # Optimization: Minimize S over theta_c, theta_o
    # We can use a grid search followed by Nelder-Mead
    best_S = float('inf')
    best_params = (0.0, 0.0)
    
    # Grid search
    theta_c_vals = np.linspace(0, 60, 61)
    theta_o_vals = np.linspace(0, 60, 61)
    
    for tc in theta_c_vals:
        for to in theta_o_vals:
            S = get_min_side(tc, to)
            if S < best_S:
                best_S = S
                best_params = (tc, to)
                
    # Refine with Nelder-Mead
    def obj(params):
        return get_min_side(params[0], params[1])
        
    try:
        res = minimize(obj, best_params, method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 1000})
        if res.fun < best_S:
            best_S = res.fun
            best_params = res.x
    except Exception:
        pass
        
    theta_c_opt, theta_o_opt = best_params
    
    # Construct final output
    # Inner hexagons:
    # We need to return (x, y, angle) for each.
    # The rigid cluster was defined with angle 0.
    # We rotated the whole cluster by theta_c_opt.
    # So each inner hexagon is rotated by theta_c_opt.
    # And its position is the centered position rotated by theta_c_opt.
    
    R_c = np.array([
        [np.cos(np.radians(theta_c_opt)), -np.sin(np.radians(theta_c_opt))],
        [np.sin(np.radians(theta_c_opt)), np.cos(np.radians(theta_c_opt))]
    ])
    
    inner = []
    for cx, cy in centered_positions:
        pos = R_c @ np.array([cx, cy])
        inner.append([pos[0], pos[1], theta_c_opt])
        
    outer_center = [0.0, 0.0]
    outer_side = best_S
    outer_angle = theta_o_opt
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
