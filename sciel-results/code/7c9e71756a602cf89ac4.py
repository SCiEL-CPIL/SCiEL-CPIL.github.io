import numpy as np
from scipy.optimize import minimize
import time

def hex_vertices(cx, cy, s, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    xs = cx + s * np.cos(angles)
    ys = cy + s * np.sin(angles)
    return np.column_stack((xs, ys))

def get_all_vertices(inner_data):
    verts = []
    for i in range(len(inner_data)):
        x, y, a = inner_data[i]
        v = hex_vertices(x, y, 1.0, a)
        verts.append(v)
    return np.vstack(verts)

def check_separation(inner_data, tol=1e-6):
    n = len(inner_data)
    hex_verts = []
    hex_normals = []
    for i in range(n):
        x, y, a = inner_data[i]
        v = hex_vertices(x, y, 1.0, a)
        hex_verts.append(v)
        normals = []
        for j in range(6):
            p1 = v[j]
            p2 = v[(j+1)%6]
            edge_vec = p2 - p1
            n_vec = np.array([-edge_vec[1], edge_vec[0]])
            n_vec /= np.linalg.norm(n_vec)
            normals.append(n_vec)
        hex_normals.append(np.array(normals))
        
    for i in range(n):
        for j in range(i+1, n):
            v1 = hex_verts[i]
            v2 = hex_verts[j]
            axes = np.vstack([hex_normals[i], hex_normals[j]])
            
            separated = False
            for axis in axes:
                proj1 = v1 @ axis
                proj2 = v2 @ axis
                
                min1 = np.min(proj1)
                max1 = np.max(proj1)
                min2 = np.min(proj2)
                max2 = np.max(proj2)
                
                if max1 < min2 - tol or max2 < min1 - tol:
                    separated = True
                    break
            
            if not separated:
                return False
                
    return True

def calculate_S(inner_data, outer_angle_deg):
    verts = get_all_vertices(inner_data)
    
    a = np.radians(outer_angle_deg)
    # Normals for outer hexagon edges
    # Base normals at 30, 90, 150 relative to 0.
    # If outer hexagon is rotated by 'a', normals are rotated by 'a'.
    base_angles = np.deg2rad([30, 90, 150])
    angles = a + base_angles
    
    norms = np.column_stack((np.cos(angles), np.sin(angles)))
    
    projs = verts @ norms.T
    max_abs_proj = np.max(np.abs(projs), axis=0)
    
    S = np.max(max_abs_proj) * 2.0 / np.sqrt(3.0)
    return S

def penalty_function(inner_data):
    n = len(inner_data)
    hex_verts = []
    hex_normals = []
    for i in range(n):
        x, y, a = inner_data[i]
        v = hex_vertices(x, y, 1.0, a)
        hex_verts.append(v)
        normals = []
        for j in range(6):
            p1 = v[j]
            p2 = v[(j+1)%6]
            edge_vec = p2 - p1
            n_vec = np.array([-edge_vec[1], edge_vec[0]])
            n_vec /= np.linalg.norm(n_vec)
            normals.append(n_vec)
        hex_normals.append(np.array(normals))
        
    penalty = 0.0
    for i in range(n):
        for j in range(i+1, n):
            v1 = hex_verts[i]
            v2 = hex_verts[j]
            axes = np.vstack([hex_normals[i], hex_normals[j]])
            
            min_overlap = 0.0
            for axis in axes:
                proj1 = v1 @ axis
                proj2 = v2 @ axis
                
                min1 = np.min(proj1)
                max1 = np.max(proj1)
                min2 = np.min(proj2)
                max2 = np.max(proj2)
                
                overlap = min(max1, max2) - max(min1, min2)
                if overlap > 0:
                    min_overlap = max(min_overlap, overlap)
            
            if min_overlap > 0:
                penalty += min_overlap ** 2
                
    return penalty

def objective(params):
    inner_data = params[:33].reshape(11, 3)
    outer_angle = params[33]
    
    S = calculate_S(inner_data, outer_angle)
    pen = penalty_function(inner_data)
    return S + 100.0 * pen

def hexagon_packing_11():
    start_time = time.time()
    deadline = 550.0
    
    d = np.sqrt(3) + 0.01
    init_inner = np.array([
        [0, 0, 0],
        [d, 0, 0],
        [d/2, d*np.sqrt(3)/2, 0],
        [-d/2, d*np.sqrt(3)/2, 0],
        [-d, 0, 0],
        [-d/2, -d*np.sqrt(3)/2, 0],
        [d/2, -d*np.sqrt(3)/2, 0],
        [2*d, 0, 0],
        [-2*d, 0, 0],
        [d, d*np.sqrt(3), 0],
        [-d, -d*np.sqrt(3), 0]
    ])
    init_outer_angle = 0.0
    
    best_params = np.concatenate([init_inner.flatten(), [init_outer_angle]])
    best_score = 1.0 / calculate_S(init_inner, init_outer_angle)
    
    try:
        x0 = best_params
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5, 'disp': False})
        
        if time.time() - start_time < deadline:
            if res.fun < objective(x0):
                candidate_inner = res.x[:33].reshape(11, 3)
                candidate_outer_angle = res.x[33]
                
                if check_separation(candidate_inner):
                    S_cand = calculate_S(candidate_inner, candidate_outer_angle)
                    score_cand = 1.0 / S_cand
                    if score_cand > best_score:
                        best_score = score_cand
                        best_params = res.x
    except Exception:
        pass
    
    final_inner = best_params[:33].reshape(11, 3)
    final_outer_angle = best_params[33]
    
    if not check_separation(final_inner):
        final_inner = init_inner
        final_outer_angle = init_outer_angle
        
    S_final = calculate_S(final_inner, final_outer_angle)
    
    outer_hex_data = np.array([0.0, 0.0, final_outer_angle])
    
    return final_inner, outer_hex_data, S_final

# EVOLVE-BLOCK-END
