import numpy as np
from scipy.optimize import minimize
import time

N = 12

def hex_vertices(cx, cy, side, angle_deg):
    """Get vertices of a regular hexagon."""
    angle = np.radians(angle_deg)
    angles = angle + np.radians(np.array([0, 60, 120, 180, 240, 300]))
    return cx + side * np.cos(angles), cy + side * np.sin(angles)

def point_in_hexagon_margin(px, py, cx, cy, side, angle_rad):
    """Return minimum distance margin: positive if inside."""
    dx = px - cx
    dy = py - cy
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    normals = np.radians(np.array([30, 90, 150, 210, 270, 330]))
    apothem = side * np.sqrt(3) / 2
    
    dots = lx * np.cos(normals) + ly * np.sin(normals)
    margins = apothem - dots
    return np.min(margins)

def get_all_inner_vertices(inner):
    """Get all vertices of all inner hexagons as arrays."""
    all_vx = []
    all_vy = []
    for x, y, ang in inner:
        vx, vy = hex_vertices(x, y, 1.0, ang)
        all_vx.append(vx)
        all_vy.append(vy)
    return np.concatenate(all_vx), np.concatenate(all_vy)

def outer_cost(params, inner_vx, inner_vy):
    """Cost = -min_margin. Minimize this to maximize min margin."""
    cx, cy, side, angle_deg = params
    if side < 0.5:
        return 1e10
    angle_rad = np.radians(angle_deg)
    
    dx = inner_vx - cx
    dy = inner_vy - cy
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    normals = np.radians(np.array([30, 90, 150, 210, 270, 330]))
    apothem = side * np.sqrt(3) / 2
    
    # For each point, compute min margin over 6 edges
    # lx[i] * cos(normals[j]) + ly[i] * sin(normals[j])
    dots = lx[:, np.newaxis] * np.cos(normals)[np.newaxis, :] + ly[:, np.newaxis] * np.sin(normals)[np.newaxis, :]
    margins = apothem - dots  # shape (n_points, 6)
    min_margin = np.min(margins)
    return -min_margin

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    sqrt3 = np.sqrt(3)
    
    # Honeycomb: 3 rows of 4 hexagons each
    inner = []
    for row in range(3):
        y_off = row * 1.5
        x_off = (row % 2) * (sqrt3 / 2)
        for col in range(4):
            inner.append([x_off + col * sqrt3, y_off, 0.0])
    
    inner_vx, inner_vy = get_all_inner_vertices(inner)
    
    # Initial guess
    cx0 = (inner_vx.min() + inner_vx.max()) / 2
    cy0 = (inner_vy.min() + inner_vy.max()) / 2
    side0 = max((inner_vx.max() - inner_vx.min()) / sqrt3, (inner_vy.max() - inner_vy.min()) / 1.5) * 1.3
    
    best_side = np.inf
    best_params = None
    
    for angle0 in [0, 30, 60, 90, 120, 150]:
        if time.time() > deadline - 20:
            break
        x0 = [cx0, cy0, side0, angle0]
        try:
            res = minimize(
                outer_cost, x0, args=(inner_vx, inner_vy),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if res.fun < best_side:
                best_side = res.fun
                best_params = res.x
        except Exception:
            pass
    
    if best_params is None:
        return inner, (float(cx0), float(cy0)), float(side0), 0.0
    
    cx, cy, side, angle = best_params
    side = max(side, 0.5)
    
    return inner, (float(cx), float(cy)), float(side), float(angle)
