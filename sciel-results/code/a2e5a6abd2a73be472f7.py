import numpy as np
from scipy.optimize import minimize
import time

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # 1. Initial Guess
    pts = []
    for m in range(-3, 4):
        for n in range(-3, 4):
            x = 2*m + n
            y = np.sqrt(3)*n
            pts.append((x, y))
            
    pts.sort(key=lambda p: p[0]**2 + p[1]**2)
    initial_pts = pts[:N]
    
    x0_pos = np.array([coord for p in initial_pts for coord in p])
    x0_phi = [0.0]
    x0 = np.concatenate([x0_pos, x0_phi])
    
    # 2. Precompute
    angles = np.deg2rad(np.arange(0, 360, 60))
    hex_verts = np.array([np.cos(angles), np.sin(angles)]).T
    
    # Base normals for outer hex (rotation 0)
    outer_angles = np.deg2rad(30 + np.arange(0, 360, 60))
    base_normals = np.array([np.cos(outer_angles), np.sin(outer_angles)]).T

    def objective(params):
        if time.time() > deadline:
            return 1e9
            
        pts_curr = params[:2*N].reshape(N, 2)
        phi = params[2*N]
        
        # Rotate outer normals
        c, s = np.cos(phi), np.sin(phi)
        R = np.array([[c, -s], [s, c]])
        normals = base_normals @ R.T
        
        # Containment
        max_proj = 0.0
        for i in range(N):
            center = pts_curr[i]
            verts = hex_verts + center
            projs = verts @ normals.T
            max_proj = max(max_proj, np.max(projs))
            
        L_req = 2.0 / np.sqrt(3.0) * max_proj
        
        # Overlap Penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pts_curr[i] - pts_curr[j])
                if d < 2.0:
                    penalty += (2.0 - d)**2
                    
        return L_req + 10.0 * penalty
        
    # 3. Optimization
    best_res = None
    best_val = np.inf
    
    for restart in range(15):
        if time.time() > deadline:
            break
            
        if restart == 0:
            x_start = x0
        else:
            x_start = x0 + np.random.normal(0, 0.5, size=x0.shape)
            x_start[2*N] = np.random.uniform(-np.pi/6, np.pi/6) # Small rotation perturbation
            
        res = minimize(objective, x_start, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4})
        
        if res.fun < best_val:
            best_val = res.fun
            best_res = res
            
        if best_val < 5.0:
            break

    if best_res is None:
        return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

    best_pts = best_res.x[:2*N].reshape(N, 2)
    best_phi = best_res.x[2*N]
    
    # Recalculate exact L
    c, s = np.cos(best_phi), np.sin(best_phi)
    R = np.array([[c, -s], [s, c]])
    normals = base_normals @ R.T
    
    max_proj = 0.0
    for i in range(N):
        center = best_pts[i]
        verts = hex_verts + center
        projs = verts @ normals.T
        max_proj = max(max_proj, np.max(projs))
        
    final_L = 2.0 / np.sqrt(3.0) * max_proj
    
    inner = [[float(x), float(y), 0.0] for x, y in best_pts]
    
    return inner, (0.0, 0.0), float(final_L), float(np.rad2deg(best_phi))
