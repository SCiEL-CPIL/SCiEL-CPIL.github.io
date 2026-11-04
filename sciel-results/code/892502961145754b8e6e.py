import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    # 1. Define the graph (3x4 grid)
    # Nodes 0..11
    # Row 0: 0, 1, 2, 3
    # Row 1: 4, 5, 6, 7
    # Row 2: 8, 9, 10, 11
    edges = []
    for i in range(3):
        for j in range(4):
            idx = i * 4 + j
            if j < 3:
                edges.append((idx, idx + 1))
            if i < 2:
                edges.append((idx, idx + 4))
    
    A = np.zeros((12, 12))
    for u, v in edges:
        A[u, v] = 1
        A[v, u] = 1
        
    D = np.diag(A.sum(axis=1))
    L = D - A
    
    # 2. Compute eigenvectors
    eigvals, eigvecs = np.linalg.eigh(L)
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Take the first two non-trivial eigenvectors (indices 1 and 2)
    x_coords = eigvecs[:, 1]
    y_coords = eigvecs[:, 2]
    
    # Center the coordinates
    x_coords -= np.mean(x_coords)
    y_coords -= np.mean(y_coords)
    
    # Initial scale estimate
    d_01 = np.sqrt((x_coords[0]-x_coords[1])**2 + (y_coords[0]-y_coords[1])**2)
    if d_01 > 1e-6:
        init_scale = 2.0 / d_01
    else:
        init_scale = 1.0
        
    # 3. Optimization
    def cost(params):
        scale, rot_in, rot_out = params
        
        # Generate centers
        cx = x_coords * scale
        cy = y_coords * scale
        
        # Check overlaps (min dist for aligned unit hexagons is 2.0)
        min_dist = np.inf
        for i in range(12):
            for j in range(i+1, 12):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < min_dist:
                    min_dist = dist
                    
        if min_dist < 2.0:
            return 1000.0 + (2.0 - min_dist) * 100.0
            
        # Calculate vertices of all inner hexagons
        angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300])) + rot_in
        rel_vx = np.cos(angles)
        rel_vy = np.sin(angles)
        
        all_vx = np.tile(cx, 6) + np.repeat(rel_vx, 12)
        all_vy = np.tile(cy, 6) + np.repeat(rel_vy, 12)
        
        # Outer hexagon normals
        normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330])) + rot_out
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        
        max_proj = 0.0
        for k in range(6):
            projections = all_vx * nx[k] + all_vy * ny[k]
            max_p = np.max(projections)
            if max_p > max_proj:
                max_proj = max_p
                
        S = 2.0 * max_proj / np.sqrt(3)
        return S

    x0 = np.array([init_scale, 0.0, 0.0])
    res = minimize(cost, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    best_scale, best_rot_in, best_rot_out = res.x
    
    cx = x_coords * best_scale
    cy = y_coords * best_scale
    
    inner = []
    for i in range(12):
        inner.append([float(cx[i]), float(cy[i]), float(np.rad2deg(best_rot_in))])
        
    center = [0.0, 0.0]
    side = float(cost(best_scale, best_rot_in, best_rot_out))
    angle = float(best_rot_out)
    
    return inner, center, side, angle
