import numpy as np
from scipy.optimize import minimize
import time

N = 11

def get_hex_vertices(center, angle_deg, side=1.0):
    cx, cy = center
    rad = np.deg2rad(angle_deg)
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300])) + rad
    verts = np.column_stack([np.cos(angles), np.sin(angles)]) * side
    return verts + np.array([cx, cy])

def point_to_segment_dist(p, a, b):
    ab = b - a
    ap = p - a
    denom = np.dot(ab, ab)
    if denom == 0:
        return np.linalg.norm(p - a)
    t = np.dot(ap, ab) / denom
    t = np.clip(t, 0, 1)
    proj = a + t * ab
    return np.linalg.norm(p - proj)

def min_dist_hex_hex(v1, v2):
    min_d = float('inf')
    for i in range(6):
        for j in range(6):
            d = np.linalg.norm(v1[i] - v2[j])
            if d < min_d: min_d = d
    for i in range(6):
        for j in range(6):
            d = point_to_segment_dist(v1[i], v2[j], v2[(j+1)%6])
            if d < min_d: min_d = d
    for i in range(6):
        for j in range(6):
            d = point_to_segment_dist(v2[i], v1[j], v1[(j+1)%6])
            if d < min_d: min_d = d
    return min_d

def objective(params):
    centers = params[:N*2].reshape(N, 2)
    angles = params[N*2:N*3]
    theta_outer = params[N*3]
    
    verts_list = []
    for i in range(N):
        v = get_hex_vertices(centers[i], angles[i])
        verts_list.append(v)
        
    rad_theta = np.deg2rad(theta_outer)
    normals = np.column_stack([np.cos(np.deg2rad(np.array([0, 60, 120, 180, 240, 300])) + rad_theta),
                                np.sin(np.deg2rad(np.array([0, 60, 120, 180, 240, 300])) + rad_theta)])
    
    max_proj = 0.0
    for v in verts_list:
        for n in normals:
            projs = np.dot(v, n)
            max_proj = max(max_proj, np.max(np.abs(projs)))
            
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d_center = np.linalg.norm(centers[i] - centers[j])
            if d_center > 2.0:
                continue
            d = min_dist_hex_hex(verts_list[i], verts_list[j])
            if d < 0.01:
                penalty += (0.01 - d) ** 2
                
    return max_proj + 100.0 * penalty

def construct():
    hex_coords = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (1,1), (-2,1), (-2,0)
    ]
    
    init_centers = []
    for q, r in hex_coords:
        x = 1.5 * q
        y = np.sqrt(3) * (r + q/2.0)
        init_centers.append([x, y])
        
    init_angles = [0.0] * N
    init_theta = 0.0
    
    x0 = np.array(init_centers).flatten()
    x0 = np.concatenate([x0, init_angles, [init_theta]])
    
    start_time = time.time()
    timeout = 110.0
    
    best_x = x0
    best_f = objective(x0)
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 3000, 'xatol': 1e-3, 'fatol': 1e-3})
        if res.fun < best_f:
            best_x = res.x
            best_f = res.fun
    except Exception:
        pass
        
    if time.time() - start_time > timeout:
        pass

    centers = best_x[:N*2].reshape(N, 2)
    angles = best_x[N*2:N*3]
    theta_outer = best_x[N*3]
    
    verts_list = []
    for i in range(N):
        v = get_hex_vertices(centers[i], angles[i])
        verts_list.append(v)
        
    rad_theta = np.deg2rad(theta_outer)
    normals = np.column_stack([np.cos(np.deg2rad(np.array([0, 60, 120, 180, 240, 300])) + rad_theta),
                                np.sin(np.deg2rad(np.array([0, 60, 120, 180, 240, 300])) + rad_theta)])
    
    max_proj = 0.0
    for v in verts_list:
        for n in normals:
            projs = np.dot(v, n)
            max_proj = max(max_proj, np.max(np.abs(projs)))
            
    L = 2.0 * max_proj / np.sqrt(3)
    
    inner = []
    for i in range(N):
        inner.append([centers[i][0], centers[i][1], angles[i]])
        
    return inner, (0.0, 0.0), L, theta_outer

# EVOLVE-BLOCK-START
# EVOLVE-BLOCK-END
