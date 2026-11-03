import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # Safety margin

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
    # Normals of the outer hexagon sides
    # Vertices at outer_angle_deg + 60k.
    # Edges between vertices k and k+1.
    # Normal to edge is at outer_angle_deg + 30 + 60k.
    normals = []
    for k in range(6):
        a = math.radians(outer_angle_deg + 30 + 60 * k)
        normals.append((math.cos(a), math.sin(a)))
    
    A = []
    b = []
    
    # Variables: [cx, cy, R] where R is circumradius of outer hex.
    # Minimize R.
    # Constraints for each vertex v and normal n:
    # -R <= (v - c) . n <= R
    # (v - c) . n <= R  =>  -c . n - R <= -v . n
    # -(v - c) . n <= R =>  c . n - R <= v . n
    
    for vx, vy in vertices:
        for nx, ny in normals:
            v_dot_n = vx * nx + vy * ny
            
            # -nx*cx - ny*cy - R <= -v_dot_n
            A.append([-nx, -ny, -1.0])
            b.append(-v_dot_n)
            
            # nx*cx + ny*cy - R <= v_dot_n
            A.append([nx, ny, -1.0])
            b.append(v_dot_n)
            
    # R >= 0
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
    except Exception:
        return None

def _calc_overlap_penalty(positions):
    """Calculate penalty for overlapping inner hexagons based on center distance."""
    penalty = 0.0
    min_dist = math.sqrt(3) # Exact minimum distance for unit hexes
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            d = math.sqrt(dx * dx + dy * dy)
            if d < min_dist:
                # Strong penalty to force separation
                penalty += (min_dist - d) ** 2 * 1000.0
    return penalty

def _objective(vars):
    """Objective function for optimizer."""
    if time.time() > _DEADLINE:
        return 1e6
    
    positions = []
    angles = []
    for i in range(N):
        x = vars[3 * i]
        y = vars[3 * i + 1]
        a = vars[3 * i + 2]
        positions.append((x, y))
        angles.append(a)
    
    # Check overlap penalty
    overlap_pen = _calc_overlap_penalty(positions)
    
    # Get vertices
    all_vertices = []
    for i in range(N):
        verts = _get_hex_vertices(positions[i][0], positions[i][1], angles[i])
        all_vertices.extend(verts)
    
    # Find min enclosing hex side (fix outer angle to 0 for optimization simplicity)
    res = _find_min_enclosing_hex_side(all_vertices, 0.0)
    if res is None:
        return 1e6 + overlap_pen * 100
    
    cx, cy, side = res
    
    # Total cost: Side length + Overlap penalty
    # Scale penalty to be comparable to side length changes
    total_cost = side + overlap_pen * 10.0
    
    return total_cost

def _generate_valid_fallback():
    """
    Generate a valid configuration using a slightly scaled triangular lattice.
    This guarantees disjointness.
    """
    sqrt3 = math.sqrt(3)
    scale = 1.01 # Slight scale up to ensure strict disjointness due to float precision
    
    # Triangular lattice vectors
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2.0 * scale, 1.5 * scale)
    
    candidates = []
    for i in range(-5, 6):
        for j in range(-5, 6):
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
    
    # Ensure we have N positions, pad if necessary (should not happen)
    while len(positions) < N:
        positions.append((1000 * len(positions), 0.0))
        
    angles = [0.0] * N
    
    all_vertices = []
    for i in range(N):
        verts = _get_hex_vertices(positions[i][0], positions[i][1], angles[i])
        all_vertices.extend(verts)
        
    res = _find_min_enclosing_hex_side(all_vertices, 0.0)
    if res:
        cx, cy, side = res
        inner = [[positions[i][0], positions[i][1], angles[i]] for i in range(N)]
        return inner, (cx, cy), side, 0.0
    else:
        # Absolute last resort: Linear chain
        inner = [[10.0 * k, 0.0, 0.0] for k in range(N)]
        return inner, (0.0, 0.0), 100.0, 0.0

def _is_valid(inner, center, side, outer_angle):
    """
    Verify that the inner hexagons are disjoint and fit inside the outer hexagon.
    """
    if len(inner) != N:
        return False
        
    # Check disjointness
    min_dist = math.sqrt(3) * 0.9999 # Tiny epsilon for floating point
    positions = [(p[0], p[1]) for p in inner]
    for i in range(N):
        for j in range(i + 1, N):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            d = math.sqrt(dx * dx + dy * dy)
            if d < min_dist:
                return False
                
    # Check containment
    # Get all vertices of inner hexagons
    all_vertices = []
    for p in inner:
        verts = _get_hex_vertices(p[0], p[1], p[2])
        all_vertices.extend(verts)
        
    # Check if all vertices are inside the outer hexagon
    # Outer hexagon center (cx, cy), side S, angle theta
    cx, cy = center
    S = side
    theta = math.radians(outer_angle)
    
    # The outer hexagon is defined by 6 inequalities:
    # (v - c) . n_k <= R_outer, where R_outer = S * sqrt(3) / 2
    R_outer = S * math.sqrt(3) / 2.0
    
    for k in range(6):
        a = theta + math.radians(30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        
        for vx, vy in all_vertices:
            val = (vx - cx) * nx + (vy - cy) * ny
            if val > R_outer + 1e-9:
                return False
                
    return True

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE:
        return _generate_valid_fallback()

    sqrt3 = math.sqrt(3)
    
    # Generate initial positions on triangular lattice (slightly scaled for safety)
    scale = 1.01
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2.0 * scale, 1.5 * scale)
    
    candidates = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            dist_sq = x * x + y * y
            candidates.append((dist_sq, x, y))
    
    candidates.sort(key=lambda t: t[0])
    
    init_positions = []
    for _, x, y in candidates:
        init_positions.append((x, y))
        if len(init_positions) == N:
            break
            
    # If we didn't get enough, pad
    while len(init_positions) < N:
        init_positions.append((100 * len(init_positions), 0.0))

    # Initial variables: x, y, angle for each hex
    x0 = []
    for i in range(N):
        x0.append(init_positions[i][0])
        x0.append(init_positions[i][1])
        x0.append(0.0) # Initial angle 0
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Run Optimization
    try:
        # Use L-BFGS-B
        res_opt = minimize(_objective, x0, method='L-BFGS-B', options={'maxiter': 300, 'maxfun': 1000})
        
        if res_opt.fun < 1000:
            best_vars = res_opt.x
            # Extract best positions
            positions = []
            angles = []
            for i in range(N):
                positions.append((best_vars[3 * i], best_vars[3 * i + 1]))
                angles.append(best_vars[3 * i + 2])
            
            # Verify and get final enclosing hex
            all_vertices = []
            for i in range(N):
                verts = _get_hex_vertices(positions[i][0], positions[i][1], angles[i])
                all_vertices.extend(verts)
            
            res_hex = _find_min_enclosing_hex_side(all_vertices, 0.0)
            if res_hex:
                cx, cy, side = res_hex
                inner_candidate = [[positions[i][0], positions[i][1], angles[i]] for i in range(N)]
                
                # STRICT Validation
                if _is_valid(inner_candidate, (cx, cy), side, 0.0):
                    if side < best_side:
                        best_side = side
                        best_inner = inner_candidate
                        best_center = (cx, cy)
                        best_outer_angle = 0.0
    except Exception:
        pass

    # Fallback: Use the pre-computed valid fallback if optimization failed or didn't improve
    fallback_inner, fallback_center, fallback_side, fallback_angle = _generate_valid_fallback()
    
    if best_inner is None or fallback_side < best_side:
        best_inner = fallback_inner
        best_center = fallback_center
        best_side = fallback_side
        best_outer_angle = fallback_angle

    return best_inner, best_center, best_side, best_outer_angle
