import numpy as np
from scipy.optimize import minimize_scalar

N = 12

def hex_vertices(cx, cy, side=1.0, angle_deg=0.0):
    angle = np.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = angle + k * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append((x, y))
    return np.array(verts)

def construct():
    sqrt3 = np.sqrt(3)
    inner = []
    # Row 0 (y=0): 4 hexagons
    for k in range(4):
        x = -1.5 * sqrt3 + k * sqrt3
        inner.append([x, 0.0, 0.0])
    # Row 1 (y=1.5): 4 hexagons, offset
    for k in range(4):
        x = -2.0 * sqrt3 + k * sqrt3
        inner.append([x, 1.5, 0.0])
    # Row 2 (y=3.0): 4 hexagons
    for k in range(4):
        x = -1.5 * sqrt3 + k * sqrt3
        inner.append([x, 3.0, 0.0])
    
    all_verts = []
    for x, y, ang in inner:
        v = hex_vertices(x, y, 1.0, ang)
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    
    def objective(theta_deg):
        theta = np.radians(theta_deg)
        normals = np.array([[np.cos(theta + k * np.pi / 3), np.sin(theta + k * np.pi / 3)] for k in range(6)])
        max_range = 0.0
        for n in normals:
            proj = all_verts @ n
            max_range = max(max_range, float(proj.max() - proj.min()))
        return max_range / np.sqrt(3)
    
    result = minimize_scalar(objective, bounds=(0, 60), method='bounded', options={'xatol': 1e-8})
    best_theta = float(result.x)
    best_s = float(result.fun)
    
    theta = np.radians(best_theta)
    normals = np.array([[np.cos(theta + k * np.pi / 3), np.sin(theta + k * np.pi / 3)] for k in range(6)])
    
    cx = 0.0
    cy = 0.0
    for n in normals:
        proj = all_verts @ n
        mid = (float(proj.max()) + float(proj.min())) / 2.0
        cx += mid * n[0]
        cy += mid * n[1]
    cx /= 6.0
    cy /= 6.0
    
    return inner, (cx, cy), best_s, best_theta
