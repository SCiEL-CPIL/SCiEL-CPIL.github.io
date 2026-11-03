"""
Hexagon packing: 11 unit hexagons inside a regular hexagon.
Uses multi-start local search with LP for outer hexagon sizing.
"""
import math
import time
import itertools
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
DEADLINE_MARGIN = 15.0  # seconds before the 150s limit

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for k in range(6):
        a = math.radians(angle_deg + 60 * k)
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return verts

def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon."""
    dx = px - cx
    dy = py - cy
    a = -math.radians(angle_deg)
    lx = dx * math.cos(a) - dy * math.sin(a)
    ly = dx * math.sin(a) + dy * math.cos(a)
    apothem = side * math.sqrt(3) / 2
    for k in range(6):
        a_norm = math.radians(30 + 60 * k)
        nx = math.cos(a_norm)
        ny = math.sin(a_norm)
        if lx * nx + ly * ny > apothem + 1e-9:
            return False
    return True

def _segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 properly intersects segment p3-p4."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

def _hexagons_disjoint(hex1, hex2, eps=1e-9):
    """Check if two hexagons have disjoint interiors."""
    cx1, cy1, s1, a1 = hex1
    cx2, cy2, s2, a2 = hex2
    r1 = s1
    r2 = s2
    dx = cx2 - cx1
    dy = cy2 - cy1
    if dx * dx + dy * dy > (r1 + r2) ** 2 + 1e-12:
        return True
    verts1 = _hex_vertices(cx1, cy1, s1, a1)
    for vx, vy in verts1:
        if _point_in_hex(vx, vy, cx2, cy2, s2, a2):
            return False
    verts2 = _hex_vertices(cx2, cy2, s2, a2)
    for vx, vy in verts2:
        if _point_in_hex(vx, vy, cx1, cy1, s1, a1):
            return False
    for i in range(6):
        v1a = verts1[i]
        v1b = verts1[(i + 1) % 6]
        for j in range(6):
            v2a = verts2[j]
            v2b = verts2[(j + 1) % 6]
            if _segments_intersect(v1a, v1b, v2a, v2b):
                return False
    return True

def _all_disjoint(inner_hexes, eps=1e-9):
    """Check if all pairs of hexagons are disjoint."""
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            if not _hexagons_disjoint(inner_hexes[i], inner_hexes[j], eps):
                return False
    return True

def _inner_in_outer(inner_hex, outer_hex):
    """Check if inner hexagon is fully inside outer hexagon."""
    cx, cy, s, a = inner_hex
    ox, oy, oside, oangle = outer_hex
    verts = _hex_vertices(cx, cy, s, a)
    for vx, vy in verts:
        if not _point_in_hex(vx, vy, ox, oy, oside, oangle):
            return False
    return True

def _compute_outer_lp(inner_hexes, oangle=0.0):
    """Find minimum side length and center of outer hexagon using LP."""
    c = [0, 0, 1]
    A_ub = []
    b_ub = []
    
    sqrt3 = math.sqrt(3)
    
    for cx_h, cy_h, s_h, ang_h in inner_hexes:
        verts = _hex_vertices(cx_h, cy_h, s_h, ang_h)
        for vx, vy in verts:
            for k in range(6):
                nx = math.cos(math.radians(oangle + 30 + 60 * k))
                ny = math.sin(math.radians(oangle + 30 + 60 * k))
                A_ub.append([-nx, -ny, -1])
                b_ub.append(-vx * nx - vy * ny)
    
    A_ub.append([0, 0, -1])
    b_ub.append(0)
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None), (None, None), (0, None)], method='highs')
    
    if res.success:
        t = res.x[2]
        side = t * 2.0 / sqrt3
        center = (res.x[0], res.x[1])
        return center, side
    else:
        return (0.0, 0.0), float('inf')

def _support_function(cx, cy, s, ang, nx, ny):
    """Support function of hexagon in direction (nx, ny)."""
    max_val = 0
    for m in range(6):
        a = math.radians(ang + 60 * m)
        vx = s * math.cos(a)
        vy = s * math.sin(a)
        val = vx * nx + vy * ny
        if val > max_val:
            max_val = val
    return max_val

def _pairwise_violation_fast(pos, angles, i, j, num_dirs=24):
    """Fast approximate violation check for optimization."""
    hex_i = (pos[2*i], pos[2*i+1], 1.0, angles[i])
    hex_j = (pos[2*j], pos[2*j+1], 1.0, angles[j])
    
    max_violation = 0
    for k in range(num_dirs):
        theta = 2.0 * math.pi * k / num_dirs
        nx = math.cos(theta)
        ny = math.sin(theta)
        
        s_i = _support_function(pos[2*i], pos[2*i+1], 1.0, angles[i], nx, ny)
        s_j = _support_function(pos[2*j], pos[2*j+1], 1.0, angles[j], -nx, -ny)
        
        dx = pos[2*j] - pos[2*i]
        dy = pos[2*j+1] - pos[2*i+1]
        dist_proj = dx * nx + dy * ny
        
        violation = s_i + s_j - dist_proj
        if violation > max_violation:
            max_violation = violation
    
    return max_violation

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    sqrt3 = math.sqrt(3)
    
    # Initial lattice configuration
    def axial_to_xy(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return (x, y)
    
    pos_axial = [(0, 0)]
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    pos_axial.extend(ring1)
    
    ring2 = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
             (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
             (0, 2), (1, 2)]
             
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_center = (0.0, 0.0)
    best_oangle = 0.0
    
    # Generate a set of initial states to try
    initial_states = []
    
    # 1. Lattice based combinations
    for combo in itertools.combinations(range(len(ring2)), 4):
        selected_r2 = [ring2[i] for i in combo]
        full_axial = pos_axial + selected_r2
        full_xy = [axial_to_xy(q, r) for q, r in full_axial]
        state = []
        for x, y in full_xy:
            state.extend([x, y, 0.0])
        initial_states.append(state)
        
    # 2. Random perturbations of the best lattice config found so far (or first one)
    if initial_states:
        base_state = initial_states[0]
        for _ in range(10):
            perturbed = base_state.copy()
            for i in range(len(perturbed)):
                if i % 3 != 2: # Perturb x and y
                    perturbed[i] += np.random.uniform(-0.2, 0.2)
            initial_states.append(perturbed)

    # Optimization function
    def objective(state):
        n_hex = len(state) // 3
        positions = []
        angles = []
        for i in range(n_hex):
            positions.append((state[3*i], state[3*i+1]))
            angles.append(state[3*i+2])
            
        inner_hexes = [(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(n_hex)]
        
        center, side = _compute_outer_lp(inner_hexes, best_oangle)
        
        if not math.isfinite(side):
            return 1e6
            
        penalty = 0
        for i in range(n_hex):
            for j in range(i + 1, n_hex):
                violation = _pairwise_violation_fast(state, angles, i, j, 24)
                if violation > 0:
                    penalty += violation ** 2
        
        return side + 100.0 * penalty

    # Multi-start local search
    for init_state in initial_states:
        if time.time() > deadline:
            break
            
        try:
            # Use Nelder-Mead for robustness
            result = minimize(objective, init_state, method='Nelder-Mead', 
                             options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
            
            if result.fun < best_side + 100:
                final_state = result.x
                n_hex = len(final_state) // 3
                hexes = []
                for i in range(n_hex):
                    hexes.append((final_state[3*i], final_state[3*i+1], 1.0, final_state[3*i+2]))
                
                if _all_disjoint(hexes):
                    inner_hexes = [(h[0], h[1], h[2], h[3]) for h in hexes]
                    center, side = _compute_outer_lp(inner_hexes, best_oangle)
                    if side < best_side:
                        best_side = side
                        best_positions = [(h[0], h[1]) for h in hexes]
                        best_angles = [h[3] for h in hexes]
                        best_center = center
        except Exception:
            pass

    if best_positions is None:
        # Fallback
        full_xy = [axial_to_xy(q, r) for q, r in pos_axial[:7]]
        best_positions = full_xy
        best_angles = [0.0] * 7
        best_center, best_side = _compute_outer_lp([(x, y, 1.0, 0.0) for x, y in best_positions], 0.0)
        best_oangle = 0.0

    # Try optimizing outer angle with the best positions found
    if time.time() < deadline - 20:
        for oangle in [0.0, 10.0, 20.0, 30.0, -10.0, -20.0, -30.0, 5.0, -5.0, 15.0, -15.0]:
            if time.time() > deadline:
                break
            inner_hexes = [(best_positions[i][0], best_positions[i][1], 1.0, best_angles[i]) for i in range(len(best_positions))]
            center, side = _compute_outer_lp(inner_hexes, oangle)
            if side < best_side:
                best_side = side
                best_center = center
                best_oangle = oangle

    # Final high-precision polish
    if time.time() < deadline - 10 and best_positions is not None:
        n_hex = len(best_positions)
        init_state = []
        for i in range(n_hex):
            init_state.extend([best_positions[i][0], best_positions[i][1], best_angles[i]])
        
        try:
            polish_result = minimize(objective, init_state, method='Nelder-Mead', 
                                     options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14})
            if polish_result.fun < best_side + 100:
                final_state = polish_result.x
                hexes = []
                for i in range(n_hex):
                    hexes.append((final_state[3*i], final_state[3*i+1], 1.0, final_state[3*i+2]))
                
                if _all_disjoint(hexes):
                    inner_hexes = [(h[0], h[1], h[2], h[3]) for h in hexes]
                    center, side = _compute_outer_lp(inner_hexes, best_oangle)
                    if side < best_side:
                        best_side = side
                        best_positions = [(h[0], h[1]) for h in hexes]
                        best_angles = [h[3] for h in hexes]
                        best_center = center
        except Exception:
            pass
    
    # Final verification
    inner_hexes = [(best_positions[i][0], best_positions[i][1], 1.0, best_angles[i]) for i in range(len(best_positions))]
    
    if not _all_disjoint(inner_hexes):
        # If invalid, try to scale up positions slightly to fix numerical errors
        for scale in [1.001, 1.005, 1.01]:
            scaled_hexes = [(x * scale, y * scale, 1.0, a) for x, y, s, a in inner_hexes]
            if _all_disjoint(scaled_hexes):
                inner_hexes = scaled_hexes
                best_positions = [(h[0], h[1]) for h in inner_hexes]
                best_angles = [h[3] for h in inner_hexes]
                best_center, best_side = _compute_outer_lp(inner_hexes, best_oangle)
                break
    
    outer_hex = (best_center[0], best_center[1], best_side, best_oangle)
    valid = True
    for h in inner_hexes:
        if not _inner_in_outer(h, outer_hex):
            valid = False
            break
    
    if not valid:
        for _ in range(100):
            best_side *= 1.001
            outer_hex = (best_center[0], best_center[1], best_side, best_oangle)
            if all(_inner_in_outer(h, outer_hex) for h in inner_hexes):
                break
    
    inner = [[inner_hexes[i][0], inner_hexes[i][1], inner_hexes[i][3]] for i in range(len(inner_hexes))]
    
    return inner, (best_center[0], best_center[1]), best_side, best_oangle
