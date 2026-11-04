import numpy as np
from scipy.optimize import minimize_scalar

N = 12

def hex_vertices(cx, cy, side, angle_deg):
    angle_rad = np.radians(angle_deg)
    base_angles = np.radians([90, 150, 210, 270, 330, 30]) + angle_rad
    return np.column_stack([cx + side * np.cos(base_angles), cy + side * np.sin(base_angles)])

def construct():
    sqrt3 = np.sqrt(3)
    positions = []
    for i in range(4):
        positions.append((i * sqrt3, 0.0))
    for i in range(3):
        positions.append((sqrt3/2 + i * sqrt3, 1.5))
    for i in range(4):
        positions.append((i * sqrt3, 3.0))
    positions.append((1.5 * sqrt3, 4.5))
    
    inner = [[x, y, 0.0] for x, y in positions]
    
    all_verts = []
    for x, y, ang in inner:
        verts = hex_vertices(x, y, 1.0, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    cx = np.mean([p[0] for p in positions])
    cy = np.mean([p[1] for p in positions])
    
    verts_centered = all_verts - np.array([cx, cy])
    
    def outer_side_length(theta_deg):
        theta_rad = np.radians(theta_deg)
        max_proj = 0.0
        for k in range(6):
            ang = theta_rad + k * np.pi / 3
            n = np.array([np.cos(ang), np.sin(ang)])
            proj = np.max(verts_centered @ n)
            if proj > max_proj:
                max_proj = proj
        return max_proj / (sqrt3 / 2.0)
    
    result = minimize_scalar(outer_side_length, bounds=(0, 60), method='bounded', options={'xatol': 1e-8})
    best_theta = result.x
    best_side = result.fun
    
    return inner, (cx, cy), best_side, best_theta
