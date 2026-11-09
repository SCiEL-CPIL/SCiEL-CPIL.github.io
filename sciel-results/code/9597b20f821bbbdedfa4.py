import numpy as np
from scipy.optimize import minimize
import time

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s buffer for 120s limit
    
    # 1. Define a valid lattice of 12 unit hexagons
    # Using a 3x4 rectangular block of pointy-top hexagons
    # Pointy-top hexagon: vertices at 0, 60, ... degrees.
    # Horizontal spacing: 2.0
    # Vertical spacing: sqrt(3)
    # Offset for odd rows: 1.0
    
    positions = []
    sqrt3 = np.sqrt(3)
    
    # 4 rows, 3 columns
    for r in range(4):
        y = r * sqrt3
        x_offset = 1.0 if (r % 2 == 1) else 0.0
        for c in range(3):
            x = c * 2.0 - 2.0 + x_offset # Centered roughly
            positions.append([x, y])
            
    # Ensure we have exactly 12
    assert len(positions) == 12
    
    # 2. Generate vertices for all inner hexagons
    # Each hexagon has side length 1.
    # Vertices are at radius 1 from center.
    all_verts = []
    base_angles = np.arange(6) * np.pi / 3
    for cx, cy in positions:
        verts = np.column_stack([
            cx + np.cos(base_angles),
            cy + np.sin(base_angles)
        ])
        all_verts.append(verts)
        
    all_verts = np.vstack(all_verts)
    
    # 3. Find the minimum enclosing regular hexagon
    # Variables: cx, cy, theta (angle of the outer hexagon)
    # The outer hexagon is defined by 6 half-planes.
    # A point p is inside if for all 6 normals n_k, |(p-c).n_k| <= S * sqrt(3)/2
    # So S >= (2/sqrt(3)) * max_k |(p-c).n_k|
    
    def objective(params):
        cx, cy, theta = params
        # Calculate normals for the 6 sides of the outer hexagon
        normal_angles = theta + np.pi/6 + np.arange(6) * np.pi / 3
        normals = np.column_stack([
            np.cos(normal_angles),
            np.sin(normal_angles)
        ])
        
        # Vectorized calculation of max projection
        shifted_verts = all_verts - np.array([cx, cy])
        
        # Dot products
        proj = np.einsum('ij,kj->ik', shifted_verts, normals)
        
        # Max absolute projection
        max_proj = np.max(np.abs(proj))
        
        # Required side length
        return max_proj * 2 / np.sqrt(3)
        
    # Initial guess
    min_x, min_y = np.min(all_verts, axis=0)
    max_x, max_y = np.max(all_verts, axis=0)
    init_c = [(min_x + max_x) / 2, (min_y + max_y) / 2]
    init_theta = 0.0
    
    best_side = float('inf')
    best_params = init_c + [init_theta]
    
    # Optimize
    # Nelder-Mead is robust for this small problem
    try:
        res = minimize(objective, init_c + [init_theta], method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
        if res.fun < best_side:
            best_side = res.fun
            best_params = res.x
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline:
        pass # Use best found so far
    
    cx, cy, theta = best_params
    side = objective(best_params)
    
    # Convert theta to degrees
    theta_deg = np.rad2deg(theta)
    
    # Prepare inner hexagons list
    # Format: (x, y, angle_degrees)
    # All inner hexagons are aligned with angle 0 in this lattice
    inner = [[x, y, 0.0] for x, y in positions]
    
    return inner, (cx, cy), side, theta_deg
