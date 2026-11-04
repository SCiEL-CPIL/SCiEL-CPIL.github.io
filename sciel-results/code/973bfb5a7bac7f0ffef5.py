import numpy as np
from scipy.optimize import minimize
import time

N = 11

def hex_vertices(cx, cy, side, angle_deg):
    theta = np.radians(angle_deg)
    angles = theta + np.radians(np.array([0, 60, 120, 180, 240, 300]))
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack([xs, ys])

def point_in_hex(px, py, cx, cy, S, phi_deg):
    dx = px - cx
    dy = py - cy
    phi = np.radians(phi_deg)
    cos_p = np.cos(-phi)
    sin_p = np.sin(-phi)
    rx = cos_p * dx - sin_p * dy
    ry = sin_p * dx + cos_p * dy
    apothem = S * np.sqrt(3) / 2
    normal_angles = np.radians(np.array([30, 90, 150, 210, 270, 330]))
    for na in normal_angles:
        val = rx * np.cos(na) + ry * np.sin(na)
        if val > apothem + 1e-9:
            return False
    return True

def objective(params, all_points):
    cx, cy, S, phi = params
    if S <= 0:
        return 1e6
    penalty = 0.0
    for px, py in all_points:
        dx = px - cx
        dy = py - cy
        phi_r = np.radians(phi)
        cos_p = np.cos(-phi_r)
        sin_p = np.sin(-phi_r)
        rx = cos_p * dx - sin_p * dy
        ry = sin_p * dx + cos_p * dy
        apothem = S * np.sqrt(3) / 2
        normal_angles = np.radians(np.array([30, 90, 150, 210, 270, 330]))
        max_viol = 0.0
        for na in normal_angles:
            val = rx * np.cos(na) + ry * np.sin(na)
            viol = val - apothem
            if viol > max_viol:
                max_viol = viol
        penalty += max_viol
    return S + 100.0 * penalty

def compute_min_hexagon(all_points, time_limit=10.0):
    start_time = time.time()
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    cx0 = (min(xs) + max(xs)) / 2.0
    cy0 = (min(ys) + max(ys)) / 2.0
    S0 = max(max(xs) - min(xs), max(ys) - min(ys)) / np.sqrt(3) + 1.0
    
    best_S = float('inf')
    best_params = None
    
    for phi0 in [0, 15, 30, 45, 60, 90]:
        if time.time() - start_time > time_limit - 1.0:
            break
        x0 = np.array([cx0, cy0, S0, phi0])
        try:
            res = minimize(objective, x0, args=(all_points,), method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x
        except Exception:
            pass
    
    return best_S, best_params

def generate_configurations():
    configs = []
    sqrt3 = np.sqrt(3)
    
    grid = {}
    for j in range(-4, 5):
        for i in range(-4, 5):
            x = i * sqrt3 + j * sqrt3 / 2
            y = j * 1.5
            grid[(i, j)] = (x, y)
    
    config_a = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
                (2,0), (1,1), (-2,1), (-1,-1)]
    pts_a = [grid[p] for p in config_a if p in grid]
    if len(pts_a) == 11:
        configs.append(pts_a)
    
    config_b = [(-1,0), (0,0), (1,0), (-2,1), (-1,1), (0,1), (1,1),
                (-1,2), (0,2), (1,2), (0,3)]
    pts_b = [grid[p] for p in config_b if p in grid]
    if len(pts_b) == 11:
        configs.append(pts_b)
    
    config_c = [(-1,-1), (0,-1), (-1,0), (0,0), (1,0),
                (-2,1), (-1,1), (0,1), (1,1), (-1,2), (0,2)]
    pts_c = [grid[p] for p in config_c if p in grid]
    if len(pts_c) == 11:
        configs.append(pts_c)
    
    config_d = [(-1,-1), (0,-1), (1,-1), (-2,0), (-1,0), (0,0), (1,0), (2,0),
                (-1,1), (0,1), (1,1)]
    pts_d = [grid[p] for p in config_d if p in grid]
    if len(pts_d) == 11:
        configs.append(pts_d)
    
    config_e = [(-2,-1), (-1,-1), (0,-1), (1,-1), (-1,0), (0,0), (1,0),
                (-2,1), (-1,1), (0,1), (1,1)]
    pts_e = [grid[p] for p in config_e if p in grid]
    if len(pts_e) == 11:
        configs.append(pts_e)
    
    config_f = [(-1,-1), (0,-1), (-2,0), (-1,0), (0,0), (1,0),
                (-1,1), (0,1), (1,1), (-1,2), (0,2)]
    pts_f = [grid[p] for p in config_f if p in grid]
    if len(pts_f) == 11:
        configs.append(pts_f)
    
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    configs = generate_configurations()
    
    best_S = float('inf')
    best_result = None
    
    for config in configs:
        if time.time() > deadline - 15.0:
            break
        
        all_points = []
        for (cx, cy) in config:
            verts = hex_vertices(cx, cy, 1.0, 0.0)
            all_points.extend(verts.tolist())
        
        S, params = compute_min_hexagon(all_points, time_limit=12.0)
        
        if S < best_S:
            best_S = S
            if params is not None:
                cx_out, cy_out, S_out, phi_out = params
                best_result = (config, (cx_out, cy_out), S_out, phi_out)
    
    if best_result is None:
        config = configs[0] if configs else [(i*np.sqrt(3), 0) for i in range(N)]
        return [[x, y, 0.0] for x, y in config], (0.0, 0.0), 100.0, 0.0
    
    config, center, S, phi = best_result
    inner = [[x, y, 0.0] for x, y in config]
    return inner, center, S, phi
