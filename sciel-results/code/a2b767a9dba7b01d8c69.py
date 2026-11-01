import numpy as np
from scipy.optimize import minimize
import time

N = 12

def hex_vertices(cx, cy, side, angle_deg):
    angle_rad = np.radians(angle_deg)
    angles = np.radians([0, 60, 120, 180, 240, 300]) + angle_rad
    return np.array([[cx + side * np.cos(a), cy + side * np.sin(a)] for a in angles])

def point_hex_violation(px, py, ox, oy, side, angle_deg):
    dx = px - ox
    dy = py - oy
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    apothem = side * np.sqrt(3) / 2
    edge_angles = np.radians([30, 90, 150, 210, 270, 330])
    max_violation = 0.0
    for ea in edge_angles:
        nx = np.cos(ea)
        ny = np.sin(ea)
        proj = lx * nx + ly * ny
        if proj > apothem:
            max_violation = max(max_violation, proj - apothem)
    return max_violation

def energy(params):
    inner = params[:N*3].reshape(N, 3)
    outer_cx, outer_cy, outer_side, outer_angle = params[N*3:]
    
    if outer_side < 0.5:
        return 1e6
    
    repulsion = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dx = inner[i,0] - inner[j,0]
            dy = inner[i,1] - inner[j,1]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = 2.0
            if dist < min_dist:
                overlap = min_dist - dist
                repulsion += 500.0 * overlap * overlap
    
    boundary_penalty = 0.0
    for i in range(N):
        verts = hex_vertices(inner[i,0], inner[i,1], 1.0, inner[i,2])
        for v in verts:
            viol = point_hex_violation(v[0], v[1], outer_cx, outer_cy, outer_side, outer_angle)
            if viol > 0:
                boundary_penalty += 100.0 * viol * viol
    
    center_penalty = 0.001 * (outer_cx**2 + outer_cy**2)
    
    return outer_side + repulsion + boundary_penalty + center_penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    positions = []
    for row in range(3):
        for col in range(4):
            x = col * 2.2 + (0.11 if row % 2 == 1 else 0)
            y = row * 1.9
            positions.append([x, y, 0.0])
    
    cx_mean = np.mean([p[0] for p in positions])
    cy_mean = np.mean([p[1] for p in positions])
    positions = [[p[0] - cx_mean, p[1] - cy_mean, p[2]] for p in positions]
    
    max_r = max(np.sqrt(p[0]**2 + p[1]**2) for p in positions) + 1.5
    initial_outer_side = max_r / (np.sqrt(3)/2) * 1.1
    
    x0 = np.array([v for p in positions for v in p] + [0.0, 0.0, initial_outer_side, 0.0])
    
    best_params = x0.copy()
    best_energy = energy(x0)
    
    n_attempts = 6
    for attempt in range(n_attempts):
        if time.time() > deadline - 40:
            break
        
        if attempt == 0:
            current = x0.copy()
        else:
            current = x0.copy()
            rng = np.random.RandomState(attempt * 12345)
            current[:N*3] += rng.normal(0, 0.8, N*3)
            current[N*3+2] = initial_outer_side * (0.85 + 0.3 * rng.rand())
        
        try:
            result = minimize(energy, current, method='L-BFGS-B', 
                            options={'maxiter': 300, 'ftol': 1e-9, 'gtol': 1e-7})
            
            if time.time() > deadline - 30:
                break
                
            if result.fun < best_energy:
                best_energy = result.fun
                best_params = result.x.copy()
                
        except Exception:
            pass
    
    inner = best_params[:N*3].reshape(N, 3)
    outer_cx, outer_cy, outer_side, outer_angle = best_params[N*3:]
    
    outer_side = max(outer_side, 1.0)
    
    inner_list = [[float(p[0]), float(p[1]), float(p[2])] for p in inner]
    
    return inner_list, (float(outer_cx), float(outer_cy)), float(outer_side), float(outer_angle)
