import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 12

def hex_vertices(x, y, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    verts = []
    for i in range(6):
        a = angle_rad + i * np.pi / 3
        verts.append((x + np.cos(a), y + np.sin(a)))
    return np.array(verts)

def min_side_for_hex(verts):
    max_val = 0
    for i in range(6):
        n = np.array([np.cos(i * np.pi / 3), np.sin(i * np.pi / 3)])
        proj = verts @ n
        val = max(abs(np.max(proj)), abs(np.min(proj)))
        if val > max_val:
            max_val = val
    return max_val / (np.sqrt(3) / 2)

def min_dist_hexes(v1, v2):
    axes = []
    for v in [v1, v2]:
        for i in range(6):
            p1 = v[i]
            p2 = v[(i+1)%6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                normal = normal / norm
                axes.append(normal)
    
    min_dist = np.inf
    for n in axes:
        proj1 = v1 @ n
        proj2 = v2 @ n
        d1 = np.max(proj1)
        d2 = np.min(proj2)
        if d1 < d2:
            dist = d2 - d1
        else:
            d1 = np.max(proj2)
            d2 = np.min(proj1)
            if d1 < d2:
                dist = d2 - d1
            else:
                dist = 0
        min_dist = min(min_dist, dist)
    return min_dist

def objective(params):
    L_req = 0
    penalty = 0
    verts_list = []
    for i in range(N):
        x = params[i*3]
        y = params[i*3+1]
        a = params[i*3+2]
        verts = hex_vertices(x, y, a)
        verts_list.append(verts)
        l_i = min_side_for_hex(verts)
        if l_i > L_req:
            L_req = l_i
            
    for i in range(N):
        for j in range(i+1, N):
            d = min_dist_hexes(verts_list[i], verts_list[j])
            if d < 0.05:
                penalty += (0.05 - d)**2
                
    return L_req + 50 * penalty

def construct():
    start_time = time.time()
    
    # Initial guess: Hex grid
    init = []
    # 12 hexes
    # Use a 3x4 grid approx
    # y = 0, -sqrt(3), -2*sqrt(3), -3*sqrt(3)
    # x shifted
    # Let's just use a circle of 11 + center
    init.append([0, 0, 0])
    for k in range(11):
        a = 2 * np.pi * k / 11
        init.append([2*np.cos(a), 2*np.sin(a), 0])
        
    x0 = np.array([v for row in init for v in row])
    
    bounds = [(-5, 5)] * (N*3)
    # Angles
    for i in range(N):
        bounds[i*3+2] = (0, 360)
        
    res = differential_evolution(objective, bounds, maxiter=30, tol=1e-3, seed=42, popsize=10)
    
    best_params = res.x
    best_val = res.fun
    
    # Polish
    if time.time() - start_time < 100:
        res2 = minimize(objective, best_params, method='Nelder-Mead', options={'maxiter': 500})
        if res2.fun < best_val:
            best_params = res2.x
            best_val = res2.fun
            
    inner = []
    for i in range(N):
        inner.append([best_params[i*3], best_params[i*3+1], best_params[i*3+2]])
        
    L = 0
    for i in range(N):
        verts = hex_vertices(inner[i][0], inner[i][1], inner[i][2])
        l_i = min_side_for_hex(verts)
        if l_i > L:
            L = l_i
            
    return inner, (0.0, 0.0), L, 0.0
