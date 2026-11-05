import numpy as np
from scipy.optimize import minimize
import time

N = 11
DEADLINE = time.time() + 140

def hex_vertices(cx, cy, side, angle_deg=0.0):
    a = np.radians(angle_deg)
    return np.array([[cx + side * np.cos(a + i * np.pi/3),
                      cy + side * np.sin(a + i * np.pi/3)] for i in range(6)])

def all_vertices(positions):
    verts = []
    for x, y, ang in positions:
        verts.append(hex_vertices(x, y, 1.0, ang))
    return np.vstack(verts)

def hex_side_containing(points, cx, cy, angle_deg):
    a = np.radians(angle_deg)
    ca, sa = np.cos(-a), np.sin(-a)
    max_proj = 0.0
    for px, py in points:
        dx, dy = px - cx, py - cy
        lx, ly = dx*ca - dy*sa, dx*sa + dy*ca
        for i in range(6):
            theta = i * np.pi / 3
            proj = lx * np.cos(theta) + ly * np.sin(theta)
            if proj > max_proj:
                max_proj = proj
    return max_proj * 2.0 / np.sqrt(3)

def bounding_hex_cost(params, points):
    cx, cy, angle = params
    return hex_side_containing(points, cx, cy, angle)

def find_bounding_hex(positions):
    points = all_vertices(positions)
    cx0 = np.mean(points[:, 0])
    cy0 = np.mean(points[:, 1])
    
    best_val = float('inf')
    best_x = None
    for ang0 in [0.0, 15.0, 30.0, 45.0, 60.0]:
        x0 = [cx0, cy0, ang0]
        try:
            res = minimize(bounding_hex_cost, x0, args=(points,), method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
    
    if best_x is None:
        return (cx0, cy0), hex_side_containing(points, cx0, cy0, 0.0), 0.0
    
    cx, cy, angle = best_x
    return (cx, cy), best_val, angle

def construct():
    d = np.sqrt(3)
    
    configs = []
    
    # 4-3-4
    p = []
    for k in range(4): p.append([k*d - 1.5*d, 0, 0])
    for k in range(3): p.append([k*d - d, d*np.sqrt(3)/2, 0])
    for k in range(4): p.append([k*d - 1.5*d, d*np.sqrt(3), 0])
    configs.append(p)
    
    # 3-5-3
    p = []
    for k in range(3): p.append([k*d - d, 0, 0])
    for k in range(5): p.append([k*d - 2*d, d*np.sqrt(3)/2, 0])
    for k in range(3): p.append([k*d - d, d*np.sqrt(3), 0])
    configs.append(p)
    
    # 1+6+4
    p = [[0, 0, 0]]
    for i in range(6):
        a = i * np.pi / 3
        p.append([d*np.cos(a), d*np.sin(a), 0])
    for i in range(4):
        a = i * np.pi / 3 + np.pi/6
        p.append([2*d*np.cos(a), 2*d*np.sin(a), 0])
    configs.append(p)
    
    best_result = None
    best_side = float('inf')
    
    for config in configs:
        if time.time() > DEADLINE:
            break
        try:
            center, side, angle = find_bounding_hex(config)
            if side < best_side:
                best_side = side
                best_result = (config, center, side, angle)
        except Exception:
            pass
    
    if best_result is None:
        config = configs[0]
        center, side, angle = find_bounding_hex(config)
        best_result = (config, center, side, angle)
    
    config, center, side, angle = best_result
    inner = [[p[0], p[1], p[2]] for p in config]
    return inner, center, side, angle
