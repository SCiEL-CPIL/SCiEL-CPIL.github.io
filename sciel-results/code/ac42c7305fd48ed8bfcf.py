"""Combined strategy: Exact lattice initialization + Robust Local Optimization + Fine Angle Sweep."""
import math
import time
import numpy as np
from scipy.optimize import minimize, linprog

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
    Use LP to find optimal center and side length for a given outer hexagon angle.
    Returns (cx, cy, side_length).
    Variables: cx, cy, t (inradius)
    Minimize t
    Constraints: (v - c) . n_k <= t for all vertices v and all 6 normals n_k
    Also: -(v - c) . n_k <= t  =>  (c - v) . n_k <= t
    """
    all_vertices = []
    for ix, iy, iangle in inner_hexes:
        verts = _get_hex_vertices(ix, iy, iangle)
        all_vertices.extend(verts)
    
    if not all_vertices:
        return 0.0, 0.0, 0.0

    n_pts = len(all_vertices)
    
    c = [0.0, 0.0, 1.0]
    A_ub = []
    b_ub = []
    
    for k in range(6):
        # Normals are perpendicular to sides. 
        # If hexagon is rotated by angle_deg, sides are at angles angle_deg + 60k.
        # Normals are at angle_deg + 30 + 60k.
        a = math.radians(angle_deg + 30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        
        # Constraint 1: (vx - cx)*nx + (vy - cy)*ny <= t
        # -nx*cx - ny*cy - t <= -nx*vx - ny*vy
        for vx, vy in all_vertices:
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-nx * vx - ny * vy)
        
        # Constraint 2: -(vx - cx)*nx - (vy - cy)*ny <= t
        # nx*cx + ny*cy - t <= nx*vx + ny*vy
        for vx, vy in all_vertices:
            A_ub.append([nx, ny, -1.0])
            b_ub.append(nx * vx + ny * vy)
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None), (None, None), (0, None)], method='highs')
        if result.success:
            cx, cy, t = result.x
            side = t / (math.sqrt(3) / 2)
            return cx, cy, side
    except Exception:
        pass
    
    # Fallback: Centroid and max projection
    cx = sum(v[0] for v in all_vertices) / n_pts
    cy = sum(v[1] for v in all_vertices) / n_pts
    required_inradius = 0.0
    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        max_proj = -float('inf')
        min_proj = float('inf')
        for vx, vy in all_vertices:
            proj = (vx - cx) * nx + (vy - cy) * ny
            max_proj = max(max_proj, proj)
            min_proj = min(min_proj, proj)
        required_inradius = max(required_inradius, max(max_proj, -min_proj))
    side = required_inradius / (math.sqrt(3) / 2)
    return cx, cy, side

def _objective(params, candidate_angles):
    """
    Objective function for optimization.
    params: flattened [x1, y1, a1, x2, y2, a2, ...]
    candidate_angles: list of outer hexagon angles to test
    Returns: minimum side length found among candidate angles
    """
    new_inner = []
    for i in range(N):
        x = params[i * 3]
        y = params[i * 3 + 1]
        a = params[i * 3 + 2]
        new_inner.append([x, y, a])
    
    # Quick disjointness check to avoid invalid states
    # Using a conservative distance check. 
    # For unit hexes, min distance is sqrt(3) if aligned. 
    # If rotated, it can be closer? No, the inradius is sqrt(3)/2. 
    # Distance between centers must be >= sqrt(3) for aligned. 
    # For arbitrary rotation, the worst case min distance is 2*inradius = sqrt(3).
    # So dist >= sqrt(3) is a necessary condition for any orientation? 
    # Actually, if one is rotated, the "width" in some direction might be larger.
    # The maximum width is 2. The minimum width is sqrt(3).
    # If centers are closer than sqrt(3), they definitely overlap.
    # If centers are between sqrt(3) and 2, they might or might not overlap depending on angle.
    # We will rely on the optimizer not going into deep overlap, but add a penalty.
    for i in range(N):
        for j in range(i + 1, N):
            dx = new_inner[i][0] - new_inner[j][0]
            dy = new_inner[i][1] - new_inner[j][1]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < math.sqrt(3) - 1e-5:
                return 1000.0 + (math.sqrt(3) - dist) * 100

    best_side = float('inf')
    for angle in candidate_angles:
        cx, cy, side = _min_enclosing_hex_side_lp(new_inner, angle)
        if side < best_side:
            best_side = side
            
    return best_side

def construct():
    global _START_TIME, _DEADLINE
    
    # Check deadline
    if time.time() > _DEADLINE - 5.0:
        # Return a safe fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # 1. Generate initial positions on a triangular lattice
    # Spacing exactly sqrt(3) for edge-to-edge contact (disjoint interiors allowed)
    sqrt3 = math.sqrt(3)
    u = (sqrt3, 0.0)
    v = (sqrt3 / 2, 1.5)
    
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
            
    # All hexes aligned at 30 degrees (pointy top) to match lattice symmetry
    # This minimizes the bounding box for this specific cluster shape
    inner = [[pos[0], pos[1], 30.0] for pos in positions]
    
    # 2. Optimization
    # We will optimize the positions and angles.
    # To speed up the objective function, we'll use a coarse set of outer angles first.
    coarse_angles = [0.0, 15.0, 30.0, 45.0, 60.0]
    
    x0 = []
    for h in inner:
        x0.extend([h[0], h[1], h[2]])
        
    best_side = float('inf')
    best_params = x0[:]
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Evaluate initial state
    init_side = _objective(x0, coarse_angles)
    if init_side < best_side:
        best_side = init_side
        # Get precise side and center for this config
        # Find best angle in coarse set
        best_angle_coarse = 0
        min_s = float('inf')
        for ang in coarse_angles:
            cx, cy, s = _min_enclosing_hex_side_lp(inner, ang)
            if s < min_s:
                min_s = s
                best_angle_coarse = ang
                best_center = (cx, cy)
        best_outer_angle = best_angle_coarse
        
    # Run optimization if time permits
    if time.time() < _DEADLINE - 40.0:
        try:
            # Use Powell's method, which is derivative-free and robust for non-smooth functions
            res = minimize(_objective, x0, args=(coarse_angles,), method='Powell',
                           options={'maxiter': 100, 'xtol': 1e-6, 'ftol': 1e-6})
            
            if res.fun < best_side:
                best_side = res.fun
                best_params = res.x
                # Update inner from best_params
                new_inner = []
                for i in range(N):
                    new_inner.append([best_params[i*3], best_params[i*3+1], best_params[i*3+2]])
                inner = new_inner
                
                # Find best angle for this new configuration
                min_s = float('inf')
                for ang in coarse_angles:
                    cx, cy, s = _min_enclosing_hex_side_lp(inner, ang)
                    if s < min_s:
                        min_s = s
                        best_angle_coarse = ang
                        best_center = (cx, cy)
                best_outer_angle = best_angle_coarse
                best_side = min_s
        except Exception as e:
            pass

    # 3. Fine Polish
    # Now that we have a good configuration, we can do a finer sweep of the outer angle
    # and a high-precision LP solve.
    if time.time() < _DEADLINE - 10.0:
        # Refine the outer angle around the best coarse angle
        fine_angles = [best_outer_angle + da for da in [-10, -5, -2, -1, 0, 1, 2, 5, 10]]
        min_s_fine = float('inf')
        best_center_fine = (0,0)
        best_angle_fine = best_outer_angle
        
        for ang in fine_angles:
            cx, cy, s = _min_enclosing_hex_side_lp(inner, ang)
            if s < min_s_fine:
                min_s_fine = s
                best_center_fine = (cx, cy)
                best_angle_fine = ang
                
        if min_s_fine < best_side:
            best_side = min_s_fine
            best_center = best_center_fine
            best_outer_angle = best_angle_fine
            
        # Even finer sweep if time allows
        if time.time() < _DEADLINE - 5.0:
            ultra_fine_angles = [best_outer_angle + da for da in [-0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5]]
            min_s_ultra = float('inf')
            for ang in ultra_fine_angles:
                cx, cy, s = _min_enclosing_hex_side_lp(inner, ang)
                if s < min_s_ultra:
                    min_s_ultra = s
                    best_center = (cx, cy)
                    best_outer_angle = ang
                    best_side = min_s_ultra

    return inner, (best_center[0], best_center[1]), best_side, best_outer_angle
