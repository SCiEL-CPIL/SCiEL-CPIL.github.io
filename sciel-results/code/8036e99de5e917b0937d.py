import numpy as np
import time
from scipy.optimize import minimize

N = 11

def get_hex_vertices(center, angle_deg, side=1.0):
    cx, cy = center
    angle_rad = np.deg2rad(angle_deg)
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack((xs, ys))

def construct():
    start_time = time.time()
    deadline = start_time + 100.0
    
    # 1. Construct the 7-hex flower
    d = np.sqrt(3)
    v1 = np.array([d, 0.0])
    v2 = np.array([d / 2.0, 1.5])
    
    # 7 centers
    centers = [
        np.array([0.0, 0.0]),
        v1, -v1,
        v2, -v2,
        v1 - v2, v2 - v1
    ]
    
    # 4 extra centers: Initialize in the "gaps" of the flower
    # Gaps are roughly at angles 30, 150, 270, 330?
    # Let's place them slightly further out in the diagonal directions
    # Direction 30 deg: (cos30, sin30)
    # Distance ~ 2.5
    extra_centers = [
        np.array([2.0, 1.2]),
        np.array([-2.0, 1.2]),
        np.array([-2.0, -1.2]),
        np.array([2.0, -1.2])
    ]
    
    all_centers = centers + extra_centers
    init_pos = np.array(all_centers)
    init_thetas = np.zeros(N)
    
    x0 = np.concatenate([init_pos.flatten(), init_thetas])
    
    # Normals for outer hexagon (aligned with inner hexes)
    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    def cost_func(params):
        if time.time() > deadline:
            return 1e9
        
        pos = params[:2*N].reshape(N, 2)
        thetas = params[2*N:]
        
        all_vertices = []
        for i in range(N):
            verts = get_hex_vertices(pos[i], thetas[i])
            all_vertices.append(verts)
        
        all_vertices = np.vstack(all_vertices)
        dots = all_vertices @ normals.T
        max_dot = np.max(dots)
        S_req = (2.0 / np.sqrt(3.0)) * max_dot
        
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(pos[i] - pos[j])
                if dist < 2.0:
                    penalty += (2.0 - dist) ** 2
        
        return S_req + 100.0 * penalty

    best_result = x0
    best_cost = cost_func(x0)
    
    # Fast Nelder-Mead
    try:
        res = minimize(cost_func, x0, method='Nelder-Mead', 
                       options={'maxiter': 200, 'fatol': 1e-5, 'xatol': 1e-5})
        if res.fun < best_cost:
            best_cost = res.fun
            best_result = res.x
    except Exception:
        pass

    pos = best_result[:2*N].reshape(N, 2)
    thetas = best_result[2*N:]
    
    all_vertices = []
    for i in range(N):
        verts = get_hex_vertices(pos[i], thetas[i])
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    dots = all_vertices @ normals.T
    max_dot = np.max(dots)
    S_final = (2.0 / np.sqrt(3.0)) * max_dot
    
    inner = []
    for i in range(N):
        inner.append([pos[i][0], pos[i][1], thetas[i]])
        
    return inner, (0.0, 0.0), S_final, 0.0

# EVOLVE-BLOCK-END
