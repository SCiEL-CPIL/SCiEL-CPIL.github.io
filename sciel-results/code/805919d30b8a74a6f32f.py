import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon."""
    angle_rad = np.radians(angle_deg)
    angles = np.arange(6) * np.pi / 3 + angle_rad
    return cx + side * np.cos(angles), cy + side * np.sin(angles)

def _compute_min_side(points, cx, cy, angle_deg):
    """Given center and rotation, find minimal circumradius to contain all points."""
    angle_rad = np.radians(angle_deg)
    max_val = 0.0
    for px, py in points:
        dx = px - cx
        dy = py - cy
        x = dx * np.cos(angle_rad) + dy * np.sin(angle_rad)
        y = -dx * np.sin(angle_rad) + dy * np.cos(angle_rad)
        for k in range(6):
            phi = np.pi / 6 + k * np.pi / 3
            val = x * np.cos(phi) + y * np.sin(phi)
            if val > max_val:
                max_val = val
    return max_val * 2.0 / np.sqrt(3)

def _objective(params, points):
    cx, cy, angle_deg = params
    return _compute_min_side(points, cx, cy, angle_deg)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    sqrt3 = np.sqrt(3)
    hex_side = 1.0
    hex_angle = 0.0
    
    # 3 rows x 4 columns hexagonal lattice
    centers = []
    for row in range(3):
        for col in range(4):
            cx = col * sqrt3 + (row % 2) * sqrt3 / 2
            cy = row * 1.5
            centers.append((cx, cy))
    
    # Center lattice at origin
    xs = [c[0] for c in centers]
    ys = [c[1] for c in centers]
    mean_x = (min(xs) + max(xs)) / 2
    mean_y = (min(ys) + max(ys)) / 2
    
    inner = []
    all_points = []
    for cx, cy in centers:
        cx_c = cx - mean_x
        cy_c = cy - mean_y
        inner.append([cx_c, cy_c, hex_angle])
        vx, vy = _hex_vertices(cx_c, cy_c, hex_side, hex_angle)
        for i in range(6):
            all_points.append((vx[i], vy[i]))
    
    all_points_arr = np.array(all_points)
    
    # Grid search for initial guess
    best_side = float('inf')
    best_params = np.array([0.0, 0.0, 0.0])
    
    for angle_try in [0.0, 30.0, -30.0, 15.0, -15.0, 45.0, -45.0, 60.0, -60.0]:
        for cx_try in np.linspace(-2, 2, 9):
            for cy_try in np.linspace(-2, 2, 9):
                side = _compute_min_side(all_points_arr, cx_try, cy_try, angle_try)
                if side < best_side:
                    best_side = side
                    best_params = np.array([cx_try, cy_try, angle_try])
    
    # Nelder-Mead refinement
    if time.time() < deadline - 40:
        try:
            res = minimize(_objective, best_params, args=(all_points_arr,), method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res.fun < best_side:
                best_params = res.x
                best_side = res.fun
        except:
            pass
    
    # High-precision polish
    if time.time() < deadline - 10:
        try:
            res = minimize(_objective, best_params, args=(all_points_arr,), method='Nelder-Mead',
                          options={'maxiter': 20000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True})
            if res.fun < best_side:
                best_params = res.x
                best_side = res.fun
        except:
            pass
    
    cx, cy, angle_deg = best_params
    side = best_side + 1e-9  # tiny safety margin
    
    return inner, (float(cx), float(cy)), float(side), float(angle_deg)
# EVOLVE-BLOCK-END
