import numpy as np
import itertools
from scipy.optimize import minimize

def construct():
    N = 12
    
    # Generate 12 centers from a hexagonal tiling
    # Pointy-top hexagons
    # Horizontal spacing: sqrt(3)
    # Vertical spacing: 2
    # Odd columns shifted by 1
    
    centers = []
    # 3 columns, 4 rows
    for i in range(-1, 2): # -1, 0, 1
        for j in range(-1, 3): # -1, 0, 1, 2
            x = i * np.sqrt(3)
            y = 2 * j + (1 if i % 2 != 0 else 0)
            centers.append([x, y])
            
    centers = np.array(centers)
    
    # Center the cluster
    centers -= np.mean(centers, axis=0)
    
    # Define the 6 vertices of a unit pointy-top hexagon
    # Angles: 90, 150, 210, 270, 330, 30
    angles = np.deg2rad([90, 150, 210, 270, 330, 30])
    verts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Function to calculate required L for a given rotation and center
    def calc_L(rotation_deg, cx, cy):
        rot = np.deg2rad(rotation_deg)
        c, s = np.cos(rot), np.sin(rot)
        R = np.array([[c, -s], [s, c]])
        
        max_L = 0
        for c_center in centers:
            # Rotate center
            rc = R @ c_center
            rc[0] += cx
            rc[1] += cy
            
            # Rotate vertices
            rv = R @ verts.T
            # Add center
            rv = rv.T + rc
            
            # Calculate L_req for each vertex
            # L = max(|y|, 2/sqrt(3)*|x|, 2/sqrt(3)*|y + x/sqrt(3)|, 2/sqrt(3)*|y - x/sqrt(3)|)
            x = rv[:, 0]
            y = rv[:, 1]
            
            L1 = np.abs(y)
            L2 = (2 / np.sqrt(3)) * np.abs(x)
            L3 = (2 / np.sqrt(3)) * np.abs(y + x / np.sqrt(3))
            L4 = (2 / np.sqrt(3)) * np.abs(y - x / np.sqrt(3))
            
            L_max = np.max(np.maximum(np.maximum(L1, L2), np.maximum(L3, L4)))
            if L_max > max_L:
                max_L = L_max
                
        return max_L

    def objective(params):
        cx, cy = params
        return calc_L(current_rot, cx, cy)

    # Optimize rotation
    best_L = float('inf')
    best_rot = 0
    best_cx = 0
    best_cy = 0
    
    # Coarse search over rotation (0 to 60 degrees due to symmetry)
    for rot in np.linspace(0, 60, 12):
        current_rot = rot
        # Initial guess for center is 0,0
        res = minimize(objective, [0, 0], method='Nelder-Mead', options={'xatol': 1e-4, 'fatol': 1e-4})
        if res.success:
            L = res.fun
            if L < best_L:
                best_L = L
                best_rot = rot
                best_cx = res.x[0]
                best_cy = res.x[1]
                
    # Fine search around best_rot
    for rot in np.linspace(best_rot - 5, best_rot + 5, 21):
        current_rot = rot
        res = minimize(objective, [best_cx, best_cy], method='Nelder-Mead', options={'xatol': 1e-5, 'fatol': 1e-5})
        if res.success:
            L = res.fun
            if L < best_L:
                best_L = L
                best_rot = rot
                best_cx = res.x[0]
                best_cy = res.x[1]

    # Final refinement
    current_rot = best_rot
    res = minimize(objective, [best_cx, best_cy], method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-6})
    if res.success:
        best_L = res.fun
        best_cx = res.x[0]
        best_cy = res.x[1]

    # Construct output
    rot_rad = np.deg2rad(best_rot)
    c, s = np.cos(rot_rad), np.sin(rot_rad)
    R = np.array([[c, -s], [s, c]])
    
    inner = []
    for c_center in centers:
        rc = R @ c_center
        rc[0] += best_cx
        rc[1] += best_cy
        inner.append([rc[0], rc[1], best_rot])
        
    return inner, (best_cx, best_cy), best_L, 0.0
