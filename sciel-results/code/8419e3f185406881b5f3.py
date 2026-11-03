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

def _logsumexp(x, beta=100.0):
    """Smooth approximation of max(x) using log-sum-exp."""
    x = np.array(x)
    x_max = np.max(x)
    if x_max == 0:
        return 0.0
    return x_max + np.log(np.sum(np.exp(beta * (x - x_max)))) / beta

def _outer_side_smooth(params, inner_positions, inner_angles, outer_angle_deg):
    """
    Smooth objective: minimize the outer hexagon side length.
    params: [cx, cy, r]
    inner_positions: list of (x, y)
    inner_angles: list of angle_degrees
    outer_angle_deg: rotation of outer hexagon
    """
    cx, cy, r = params
    
    # Compute all vertex projections onto the 6 normal directions
    projections = []
    for k in range(6):
        a = math.radians(outer_angle_deg + 30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        for ix, iy in inner_positions:
            for iangle in inner_angles:
                verts = _get_hex_vertices(ix, iy, iangle)
                for vx, vy in verts:
                    proj = (vx - cx) * nx + (vy - cy) * ny
                    projections.append(proj)
    
    projections = np.array(projections)
    # Smooth max of all projections
    r_required = _logsumexp(projections, beta=50.0)
    
    # Objective: minimize r (inradius), which is proportional to side length
    return r_required / (math.sqrt(3) / 2)

def _objective_smooth(params):
    """Objective function for smooth optimization."""
    if time.time() > _DEADLINE - 2.0:
        return 1e9
    try:
        cx, cy, r = params
        return _outer_side_smooth([cx, cy, r], _global_inner_positions, _global_inner_angles, _global_outer_angle)
    except Exception:
        return 1e9

_global_inner_positions = []
_global_inner_angles = []
_global_outer_angle = 0.0

def _find_best_outer_smooth(inner_hexes):
    """
    Find the best outer hexagon using smooth optimization.
    """
    global _global_inner_positions, _global_inner_angles, _global_outer_angle
    
    _global_inner_positions = [(h[0], h[1]) for h in inner_hexes]
    _global_inner_angles = [h[2] for h in inner_hexes]
    
    best_side = float('inf')
    best_angle = 0.0
    best_center = (0.0, 0.0)
    
    # Coarse sweep over outer angles
    for angle_deg in np.arange(0, 60, 2.0):
        _global_outer_angle = float(angle_deg)
        
        # Initial guess: centroid of inner positions
        cx0 = sum(p[0] for p in _global_inner_positions) / len(_global_inner_positions)
        cy0 = sum(p[1] for p in _global_inner_positions) / len(_global_inner_positions)
        
        x0 = np.array([cx0, cy0, 0.0])
        
        res = minimize(_objective_smooth, x0, method='Nelder-Mead',
                       options={'xatol': 1e-5, 'fatol': 1e-7, 'maxiter': 300, 'adaptive': True})
        
        if res.fun < best_side:
            best_side = res.fun
            best_angle = float(angle_deg)
            cx, cy, r = res.x
            best_center = (cx, cy)
    
    # Refine around best angle with finer sweep
    for angle_deg in np.arange(best_angle - 3, best_angle + 3, 0.5):
        if angle_deg < 0 or angle_deg >= 60:
            continue
        _global_outer_angle = float(angle_deg)
        
        cx0 = best_center[0]
        cy0 = best_center[1]
        
        x0 = np.array([cx0, cy0, 0.0])
        
        res = minimize(_objective_smooth, x0, method='Nelder-Mead',
                       options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 500, 'adaptive': True})
        
        if res.fun < best_side:
            best_side = res.fun
            best_angle = float(angle_deg)
            cx, cy, r = res.x
            best_center = (cx, cy)
    
    # Final LP verification for exact side length
    _global_outer_angle = best_angle
    cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, best_angle)
    
    return cx, cy, side, best_angle

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
    
    cx, cy, side, angle = _find_best_outer_smooth(inner)
    
    return inner, (cx, cy), side, angle
# EVOLVE-BLOCK-END
