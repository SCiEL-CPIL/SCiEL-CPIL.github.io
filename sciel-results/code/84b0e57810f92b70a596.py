"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # 5s safety margin

def _get_hex_vertices(cx, cy, angle_deg=0.0):
    """Get vertices of a unit regular hexagon centered at (cx, cy) with rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _min_enclosing_hex_side_lp(inner_hexes, angle_deg):
    """
    For a fixed outer hexagon angle, find optimal center and side length using LP.
    Returns (cx, cy, side, angle_deg).
    """
    all_vertices = []
    for ix, iy, iangle in inner_hexes:
        verts = _get_hex_vertices(ix, iy, iangle)
        all_vertices.extend(verts)
    
    n_pts = len(all_vertices)
    if n_pts == 0:
        return 0.0, 0.0, 0.0, angle_deg

    A_ub = []
    b_ub = []
    
    # Outer hexagon normals for given angle
    # The constraint is that all vertices must be inside the outer hexagon.
    # Outer hexagon with inradius r, center (cx, cy), rotation angle_deg.
    # Normal directions are angle_deg + 30 + 60*k.
    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        for vx, vy in all_vertices:
            # (vx - cx)*nx + (vy - cy)*ny <= r
            # -nx*cx - ny*cy - r <= -vx*nx - vy*ny
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-vx * nx - vy * ny)
    
    # r >= 0
    A_ub.append([0.0, 0.0, -1.0])
    b_ub.append(0.0)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    c = np.array([0.0, 0.0, 1.0])
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs', options={'presolve': True})
        if result.success:
            cx_opt = result.x[0]
            cy_opt = result.x[1]
            r_opt = result.x[2]
            side = r_opt / (math.sqrt(3) / 2)
            return cx_opt, cy_opt, side, angle_deg
    except Exception:
        pass

    # Fallback: Centroid based
    cx_mean = sum(v[0] for v in all_vertices) / n_pts
    cy_mean = sum(v[1] for v in all_vertices) / n_pts
    r_opt = 0.0
    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        for vx, vy in all_vertices:
            proj = (vx - cx_mean) * nx + (vy - cy_mean) * ny
            r_opt = max(r_opt, proj)
    side = r_opt / (math.sqrt(3) / 2)
    return cx_mean, cy_mean, side, angle_deg

def _objective(angle_params):
    """Objective function for Nelder-Mead: minimize side length for a given angle."""
    angle_deg = angle_params[0]
    if time.time() > _DEADLINE - 2.0:
        return 1e9
    try:
        _, _, side, _ = _min_enclosing_hex_side_lp(_global_inner_hexes, angle_deg)
        return side
    except Exception:
        return 1e9

_global_inner_hexes = []

def _find_best_outer_lp(inner_hexes):
    """
    Find the best outer hexagon by optimizing the angle using Nelder-Mead.
    """
    global _global_inner_hexes
    _global_inner_hexes = inner_hexes
    
    # Initial coarse sweep to find a good starting point
    best_side = float('inf')
    best_angle = 0.0
    best_center = (0.0, 0.0)
    
    angles_coarse = np.arange(0, 60, 1.0)
    for angle_deg in angles_coarse:
        cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg))
        if side < best_side:
            best_side = side
            best_center = (cx, cy)
            best_angle = float(angle_deg)
            
    # Nelder-Mead refinement on angle
    # We optimize over [best_angle - 5, best_angle + 5] effectively, 
    # but Nelder-Mead will handle the local landscape.
    # Note: The function is periodic with period 60.
    # We can restrict the search to a small window around the best coarse angle.
    
    x0 = np.array([best_angle])
    
    # Bounds are not directly supported by Nelder-Mead, but we can penalize out-of-bounds
    # or rely on the fact that the coarse sweep found a local min.
    # The landscape is smooth.
    
    res = minimize(_objective, x0, method='Nelder-Mead', 
                   options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 200, 'adaptive': True})
    
    if res.success and res.fun < best_side:
        best_angle = res.x[0]
        # Get the final center and side
        cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, best_angle)
        best_side = side
        best_center = (cx, cy)
    else:
        # Use the best from coarse sweep
        cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, best_angle)
        best_side = side
        best_center = (cx, cy)

    return best_center[0], best_center[1], best_side, best_angle

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    sqrt3 = math.sqrt(3)
    scale = 1.001
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)
    
    candidates = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            dist_sq = x * x + y * y
            candidates.append((dist_sq, x, y))
    
    candidates.sort(key=lambda t: t[0])
    
    positions = []
    for _, x, y in candidates:
        positions.append((x, y))
        if len(positions) == N:
            break
    
    inner = [[pos[0], pos[1], 30.0] for pos in positions]
    
    if time.time() > _DEADLINE - 10.0:
        xs = [h[0] for h in inner]
        ys = [h[1] for h in inner]
        cx = (max(xs) + min(xs)) / 2
        cy = (max(ys) + min(ys)) / 2
        max_dist = 0
        for h in inner:
            d = math.sqrt((h[0] - cx)**2 + (h[1] - cy)**2)
            if d > max_dist:
                max_dist = d
        side = (max_dist + 1.0) / (math.sqrt(3) / 2) + 0.1
        return inner, (cx, cy), side, 0.0
    
    cx, cy, side, angle = _find_best_outer_lp(inner)
    
    return inner, (cx, cy), side, angle
# EVOLVE-BLOCK-END
