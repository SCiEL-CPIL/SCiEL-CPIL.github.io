import math
import time
import numpy as np
from scipy.optimize import linprog, minimize, differential_evolution

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 120.0

def _get_hex_vertices(cx, cy, angle_deg=0.0):
    """Get vertices of a unit regular hexagon centered at (cx, cy) with rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _find_min_enclosing_hex_side(vertices, outer_angle_deg):
    """
    Find the minimum side length of a regular hexagon with given angle
    that contains all given vertices.
    Returns (center_x, center_y, side_length) or None.
    """
    normals = []
    for k in range(6):
        a = math.radians(outer_angle_deg + 30 + 60 * k)
        normals.append((math.cos(a), math.sin(a)))
    
    A = []
    b = []
    
    for vx, vy in vertices:
        for nx, ny in normals:
            v_dot_n = vx * nx + vy * ny
            
            A.append([-nx, -ny, -1.0])
            b.append(-v_dot_n)
            
            A.append([nx, ny, -1.0])
            b.append(v_dot_n)
            
    A.append([0.0, 0.0, -1.0])
    b.append(0.0)
    
    c = [0.0, 0.0, 1.0]
    
    try:
        res = linprog(c, A_ub=A, b_ub=b, bounds=[(None, None), (None, None), (0, None)], method='highs')
        if res.success:
            cx, cy, R = res.x
            side = R / (math.sqrt(3) / 2)
            return cx, cy, side
        else:
            return None
    except:
        return None

def _smooth_outer_side(all_vertices, outer_angle_deg):
    """
    Smooth surrogate for the minimum enclosing hexagon side length.
    Uses the exact LP result as the objective value.
    """
    res = _find_min_enclosing_hex_side(all_vertices, outer_angle_deg)
    if res is None:
        return 1e10
    return res[2]

def _polish_with_lbfgsb(all_vertices, init_center, init_angle, init_side, max_iter=1000):
    """
    Polish the solution using L-BFGS-B with the exact LP-based objective.
    The LP gives the optimal center for a given angle, so we only need to optimize angle.
    """
    def objective(angle_params):
        angle_deg = angle_params[0]
        res = _find_min_enclosing_hex_side(all_vertices, angle_deg)
        if res is None:
            return 1e10
        return res[2]
    
    x0 = np.array([init_angle])
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', 
                         options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-15})
        if result.success:
            angle_deg = result.x[0]
            res = _find_min_enclosing_hex_side(all_vertices, angle_deg)
            if res:
                return res
    except Exception:
        pass
    
    return None

def _generate_hex_layout(n_hexes):
    """
    Generate a hexagonal arrangement of n unit hexagons.
    Returns list of (x, y) positions.
    """
    sqrt3 = math.sqrt(3)
    scale = 1.000001  # Slightly larger than touching
    
    # Hexagonal lattice vectors
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)
    
    # Generate candidates in a hexagonal pattern
    candidates = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            dist_sq = x * x + y * y
            candidates.append((dist_sq, x, y))
    
    candidates.sort(key=lambda t: t[0])
    
    positions = []
    for _, x, y in candidates:
        positions.append((x, y))
        if len(positions) == n_hexes:
            break
    
    return positions

def _try_layout(positions, inner_angle, deadline_margin=15.0):
    """
    Try a given layout and inner angle, optimize outer hexagon.
    Returns (side, center, outer_angle) or None.
    """
    all_vertices = []
    for px, py in positions:
        verts = _get_hex_vertices(px, py, inner_angle)
        all_vertices.extend(verts)
    
    best_coarse_side = float('inf')
    best_coarse_angle = 0.0
    best_coarse_center = (0.0, 0.0)
    
    # Coarse search for outer angle
    for outer_angle_deg in np.arange(0, 30.1, 0.5):
        if time.time() > _DEADLINE - deadline_margin:
            break
        res = _find_min_enclosing_hex_side(all_vertices, outer_angle_deg)
        if res:
            cx, cy, side = res
            if side < best_coarse_side:
                best_coarse_side = side
                best_coarse_angle = outer_angle_deg
                best_coarse_center = (cx, cy)
    
    # Fine search around best coarse angle
    if best_coarse_angle >= 0 and best_coarse_side < float('inf'):
        search_range = 0.5
        step = 0.01
        for outer_angle_deg in np.arange(max(0, best_coarse_angle - search_range), 
                                        min(30, best_coarse_angle + search_range) + step, step):
            if time.time() > _DEADLINE - deadline_margin:
                break
            res = _find_min_enclosing_hex_side(all_vertices, outer_angle_deg)
            if res:
                cx, cy, side = res
                if side < best_coarse_side:
                    best_coarse_side = side
                    best_coarse_angle = outer_angle_deg
                    best_coarse_center = (cx, cy)
    
    # Polish with L-BFGS-B if time permits
    if time.time() < _DEADLINE - deadline_margin and best_coarse_side < float('inf'):
        polished = _polish_with_lbfgsb(all_vertices, best_coarse_center, 
                                       best_coarse_angle, best_coarse_side, max_iter=500)
        if polished:
            cx, cy, side = polished
            if side < best_coarse_side:
                best_coarse_side = side
                best_coarse_center = (cx, cy)
                # Re-find angle from polished result
                # The polished result already has the optimal angle from LP
    
    if best_coarse_side < float('inf'):
        return best_coarse_side, best_coarse_center, best_coarse_angle
    
    return None

def _generate_alternative_layouts(n_hexes):
    """
    Generate multiple alternative hexagonal layouts for n unit hexagons.
    Returns list of lists of (x, y) positions.
    """
    sqrt3 = math.sqrt(3)
    scale = 1.000001
    
    layouts = []
    
    # Standard hexagonal packing (1-3-5-3-1 pattern for 11)
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)
    
    # Layout 1: Standard hexagonal arrangement
    candidates = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            dist_sq = x * x + y * y
            candidates.append((dist_sq, x, y))
    
    candidates.sort(key=lambda t: t[0])
    layout1 = [(x, y) for _, x, y in candidates[:n_hexes]]
    layouts.append(layout1)
    
    # Layout 2: Linear arrangement
    layout2 = [(i * sqrt3 * scale, 0.0) for i in range(n_hexes)]
    layouts.append(layout2)
    
    # Layout 3: Two rows
    layout3 = []
    for i in range(n_hexes):
        row = i // 6
        col = i % 6
        x = col * sqrt3 * scale + row * sqrt3 / 2 * scale
        y = row * 1.5 * scale
        layout3.append((x, y))
    layouts.append(layout3)
    
    return layouts

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Generate multiple layouts
    layouts = _generate_alternative_layouts(N)
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    best_layout_idx = 0
    best_inner_angle = 0.0
    
    # Try different inner angles
    inner_angles = [0.0, 30.0]
    
    for layout_idx, positions in enumerate(layouts):
        if time.time() > _DEADLINE - 20.0:
            break
        
        for inner_angle in inner_angles:
            if time.time() > _DEADLINE - 20.0:
                break
            
            result = _try_layout(positions, inner_angle, deadline_margin=20.0)
            if result:
                side, center, outer_angle = result
                if side < best_side:
                    best_side = side
                    best_inner = [[pos[0], pos[1], inner_angle] for pos in positions]
                    best_center = center
                    best_outer_angle = outer_angle
                    best_layout_idx = layout_idx
                    best_inner_angle = inner_angle
    
    if best_inner is None:
        # Fallback if optimization failed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, best_center, best_side, best_outer_angle
