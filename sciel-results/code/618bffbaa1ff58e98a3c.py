import numpy as np
from scipy.optimize import minimize
import time

def hexagon_packing_11():
    # Normals for a hexagon with angle 0
    # Normals at 30, 90, 150, 210, 270, 330 degrees
    angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Vertices of a unit hexagon (angle 0)
    v_angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    vertices = np.column_stack([np.cos(v_angles), np.sin(v_angles)])
    
    # Initial configuration: 7-flower + 4 extra
    centers = np.array([
        [0, 0],
        [np.sqrt(3), 0],
        [np.sqrt(3)/2, 1.5],
        [-np.sqrt(3)/2, 1.5],
        [-np.sqrt(3), 0],
        [-np.sqrt(3)/2, -1.5],
        [np.sqrt(3)/2, -1.5]
    ])
    extra_angles = np.deg2rad([30, 150, 210, 330])
    extra_centers = 2 * np.sqrt(3) * np.column_stack([np.cos(extra_angles), np.sin(extra_angles)])
    centers = np.vstack([centers, extra_centers])
    
    def objective(params):
        c = params.reshape(11, 2)
        
        # Calculate required S
        max_proj = 0
        for i in range(11):
            pts = c[i] + vertices
            projs = pts @ normals.T
            max_proj = max(max_proj, np.max(np.abs(projs)))
        
        S_req = max_proj * 2 / np.sqrt(3)
        
        # Calculate overlap penalty
        penalty = 0
        for i in range(11):
            for j in range(i+1, 11):
                d = c[i] - c[j]
                projs = np.abs(d @ normals.T)
                min_gap = np.max(projs) - np.sqrt(3)
                if min_gap < 0:
                    penalty += (-min_gap) ** 2
        
        return S_req + 100 * penalty

    best_S = np.inf
    best_params = centers.flatten()
    
    start_time = time.time()
    deadline = 550  # Leave 50s for overhead
    
    # Run restarts
    for restart in range(10):
        if time.time() > deadline:
            break
            
        if restart == 0:
            x0 = centers.flatten()
        else:
            x0 = centers.flatten() + np.random.normal(0, 0.2, 22)
            
        # Use Nelder-Mead
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5})
        
        # Check validity
        c = res.x.reshape(11, 2)
        is_valid = True
        for i in range(11):
            for j in range(i+1, 11):
                d = c[i] - c[j]
                projs = np.abs(d @ normals.T)
                if np.max(projs) < np.sqrt(3) - 1e-6:
                    is_valid = False
                    break
            if not is_valid:
                break
        
        if is_valid:
            # Calculate S
            max_proj = 0
            for i in range(11):
                pts = c[i] + vertices
                projs = pts @ normals.T
                max_proj = max(max_proj, np.max(np.abs(projs)))
            S_val = max_proj * 2 / np.sqrt(3)
            
            if S_val < best_S:
                best_S = S_val
                best_params = res.x

    c = best_params.reshape(11, 2)
    
    # Final S calculation
    max_proj = 0
    for i in range(11):
        pts = c[i] + vertices
        projs = pts @ normals.T
        max_proj = max(max_proj, np.max(np.abs(projs)))
    
    S = max_proj * 2 / np.sqrt(3)
    S += 1e-7  # Safety margin
    
    inner_hex_data = np.column_stack([c, np.zeros(11)])
    outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, S

hexagon_packing_11()
