"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import linprog

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 120.0  # 120s safety margin within 150s limit

def _get_hex_vertices(cx, cy, angle_deg=0.0):
    """Get vertices of a unit regular hexagon centered at (cx, cy) with rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _min_enclosing_hex_side_lp(inner_hexes, angle_deg):
    """Use LP to find optimal center and inradius for given outer hexagon angle.
    Returns (cx, cy, side_length) or None on failure."""
    all_vertices = []
    for ix, iy, iangle in inner_hexes:
        verts = _get_hex_vertices(ix, iy, iangle)
        all_vertices.extend(verts)
    
    n_pts = len(all_vertices)
    
    # Variables: [cx, cy, t] where t is the inradius (half-width in normal direction)
    # Minimize t
    c = np.array([0.0, 0.0, 1.0])
    
    A_ub = []
    b_ub = []
    
    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        
        # For each vertex: (vx - cx)*nx + (vy - cy)*ny <= t
        # => -nx*cx - ny*cy - t <= -(nx*vx + ny*vy)
        for vx, vy in all_vertices:
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-(nx * vx + ny * vy))
        
        # For each vertex: -(vx - cx)*nx - (vy - cy)*ny <= t
        # => nx*cx + ny*cy - t <= nx*vx + ny*vy
        for vx, vy in all_vertices:
            A_ub.append([nx, ny, -1.0])
            b_ub.append(nx * vx + ny * vy)
    
    # t >= 0
    A_ub.append([0.0, 0.0, -1.0])
    b_ub.append(0.0)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, 
                      bounds=[(None, None), (None, None), (0, None)], 
                      method='highs')
        if res.success:
            cx, cy, t = res.x
            side = t / (math.sqrt(3) / 2)
            return cx, cy, side
        else:
            return None
    except Exception:
        return None

def construct():
    global _START_TIME, _DEADLINE
    
    # Check deadline
    if time.time() > _DEADLINE - 5.0:
        # Return a safe fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Generate 11 points in a hexagonal cluster arrangement
    # Central hexagon + ring of 6 + 4 more in second ring
    # Use triangular lattice with spacing sqrt(3) for edge-to-edge contact
    sqrt3 = math.sqrt(3)
    # Scale factor to ensure strict disjointness (small gap between hexagons)
    scale = 1.001
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)
    
    # Generate candidate points in a spiral/grid around the origin
    # We want the 11 closest points to the origin in the triangular lattice
    candidates = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            dist_sq = x * x + y * y
            candidates.append((dist_sq, x, y))
    
    # Sort by distance
    candidates.sort(key=lambda t: t[0])
    
    # Take the first N unique points
    positions = []
    for _, x, y in candidates:
        positions.append((x, y))
        if len(positions) == N:
            break
    
    # Create inner hexagons with all angles = 30 (Pointy-Top)
    inner = [[pos[0], pos[1], 30.0] for pos in positions]
    
    # Compute minimum enclosing hexagon using LP
    if time.time() > _DEADLINE - 10.0:
        # Quick estimate fallback
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
    
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Coarse search over outer hexagon angles (0 to 30 degrees, exploiting symmetry)
    coarse_angles = np.arange(0, 30.5, 0.5)
    for angle_deg in coarse_angles:
        if time.time() > _DEADLINE - 10.0:
            break
        result = _min_enclosing_hex_side_lp(inner, angle_deg)
        if result is not None:
            cx, cy, side = result
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = angle_deg
    
    # Fine search around the best coarse angle
    if best_angle > 0 and time.time() < _DEADLINE - 10.0:
        search_range = 0.5
        step = 0.01
        fine_angles = np.arange(max(0, best_angle - search_range), 
                                min(30, best_angle + search_range) + step, step)
        for angle_deg in fine_angles:
            if time.time() > _DEADLINE - 10.0:
                break
            result = _min_enclosing_hex_side_lp(inner, angle_deg)
            if result is not None:
                cx, cy, side = result
                if side < best_side:
                    best_side = side
                    best_center = (cx, cy)
                    best_angle = angle_deg
    
    # Final ultra-fine polish
    if time.time() < _DEADLINE - 5.0:
        search_range = 0.05
        step = 0.001
        fine_angles = np.arange(max(0, best_angle - search_range), 
                                min(30, best_angle + search_range) + step, step)
        for angle_deg in fine_angles:
            if time.time() > _DEADLINE - 5.0:
                break
            result = _min_enclosing_hex_side_lp(inner, angle_deg)
            if result is not None:
                cx, cy, side = result
                if side < best_side:
                    best_side = side
                    best_center = (cx, cy)
                    best_angle = angle_deg
    
    return inner, best_center, best_side, best_angle
# EVOLVE-BLOCK-END
