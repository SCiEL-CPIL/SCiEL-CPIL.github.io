"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12
import numpy as np
from scipy.optimize import minimize
import time

# EVOLVE-BLOCK-START
def _get_hex_verts(cx, cy, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300])) + angle_rad
    x = np.cos(angles)
    y = np.sin(angles)
    return np.column_stack((x + cx, y + cy))

def _objective(params):
    inner_params = params[:3*N]
    cx = params[3*N]
    cy = params[3*N+1]
    phi = params[3*N+2]
    
    all_verts = []
    centers = []
    for i in range(N):
        x = inner_params[3*i]
        y = inner_params[3*i+1]
        a = inner_params[3*i+2]
        verts = _get_hex_verts(x, y, a)
        all_verts.append(verts)
        centers.append(np.array([x, y]))
        
    all_verts = np.vstack(all_verts)
    centers = np.array(centers)
    
    phi_rad = np.deg2rad(phi)
    normals_angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + phi)
    normals = np.column_stack((np.cos(normals_angles), np.sin(normals_angles)))
    
    projections = np.abs(all_verts @ normals.T)
    max_proj = np.max(projections)
    
    R = max_proj
    L_req = 2 * R / np.sqrt(3)
    
    penalty = 0.0
    min_dist = np.sqrt(3)
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_dist:
                penalty += (min_dist - d)**2
                
    return L_req + 1000.0 * penalty

def construct():
    start_time = time.time()
    deadline = start_time + 100.0
    
    best_L = float('inf')
    best_params = None
    
    rng = np.random.default_rng(42)
    num_restarts = 10
    
    for k in range(num_restarts):
        if time.time() > deadline:
            break
            
        if k == 0:
            init_params = []
            init_params.extend([0.0, 0.0, 0.0])
            for i in range(6):
                ang = np.deg2rad(i * 60)
                x = np.cos(ang) * np.sqrt(3)
                y = np.sin(ang) * np.sqrt(3)
                init_params.extend([x, y, 0.0])
            for i in range(5):
                ang = np.deg2rad(i * 72)
                x = np.cos(ang) * 3.5
                y = np.sin(ang) * 3.5
                init_params.extend([x, y, 0.0])
            init_params.extend([0.0, 0.0, 0.0])
            x0 = np.array(init_params)
        else:
            x0 = rng.uniform(-4, 4, 3*N + 3)
            x0[-1] = 0.0
            x0[-2] = 0.0
            x0[-3] = 0.0

        res = minimize(_objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
        
        if res.fun < best_L:
            best_L = res.fun
            best_params = res.x
            
    if best_params is not None:
        res = minimize(_objective, best_params, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5})
        if res.fun < best_L:
            best_L = res.fun
            best_params = res.x

    if best_params is None:
        inner = [[4.0 * i, 0.0, 0.0] for i in range(N)]
        return inner, (0.0, 0.0), 40.0, 0.0

    inner_params = best_params[:3*N]
    cx = best_params[3*N]
    cy = best_params[3*N+1]
    phi = best_params[3*N+2]
    
    inner = []
    for i in range(N):
        x = inner_params[3*i]
        y = inner_params[3*i+1]
        a = inner_params[3*i+2]
        inner.append([x, y, a])
        
    return inner, (cx, cy), best_L, phi
# EVOLVE-BLOCK-END
