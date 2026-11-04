"""Packing 12 unit regular hexagons inside a minimal regular hexagon."""
import math
import time
import numpy as np

N = 12
DEADLINE = 135.0  # safety margin before 150s kill

def _hex_vertices(cx, cy, side, rot_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    rot = math.radians(rot_deg)
    verts = []
    for i in range(6):
        angle = rot + i * math.pi / 3.0
        x = cx + side * math.cos(angle)
        y = cy + side * math.sin(angle)
        verts.append((x, y))
    return verts

def _point_in_hex(px, py, cx, cy, side, rot_deg):
    """Check if point (px,py) is inside hexagon with center (cx,cy), side, rotation."""
    # Transform point to hexagon's local frame
    dx = px - cx
    dy = py - cy
    rot = math.radians(-rot_deg)
    lx = dx * math.cos(rot) - dy * math.sin(rot)
    ly = dx * math.sin(rot) + dy * math.cos(rot)
    
    # For a regular hexagon with side s, the distance from center to each side is s*sqrt(3)/2
    # The hexagon in local coords has vertices at angles 0, 60, 120, 180, 240, 300 degrees
    # A point is inside if it satisfies all 6 half-plane constraints
    apothem = side * math.sqrt(3) / 2.0
    for i in range(6):
        angle = i * math.pi / 3.0
        # Normal pointing inward
        nx = math.cos(angle)
        ny = math.sin(angle)
        if lx * nx + ly * ny > apothem:
            return False
    return True

def _hex_distance(h1, h2):
    """
    h1 = (cx1, cy1, angle1), h2 = (cx2, cy2, angle2)
    Returns minimum distance between the two unit hexagons (0 if overlapping or touching).
    Uses vertex-edge and edge-edge distance checks.
    """
    v1 = _hex_vertices(h1[0], h1[1], 1.0, h1[2])
    v2 = _hex_vertices(h2[0], h2[1], 1.0, h2[2])
    
    min_dist = float('inf')
    
    # Check vertex of h1 against edges of h2
    for vx, vy in v1:
        for i in range(6):
            ex1, ey1 = v2[i]
            ex2, ey2 = v2[(i + 1) % 6]
            d = _point_segment_distance(vx, vy, ex1, ey1, ex2, ey2)
            min_dist = min(min_dist, d)
    
    # Check vertex of h2 against edges of h1
    for vx, vy in v2:
        for i in range(6):
            ex1, ey1 = v1[i]
            ex2, ey2 = v1[(i + 1) % 6]
            d = _point_segment_distance(vx, vy, ex1, ey1, ex2, ey2)
            min_dist = min(min_dist, d)
    
    return min_dist

def _point_segment_distance(px, py, ax, ay, bx, by):
    """Distance from point (px,py) to line segment from (ax,ay) to (bx,by)."""
    dx = bx - ax
    dy = by - ay
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-12:
        return math.sqrt((px - ax) ** 2 + (py - ay) ** 2)
    t = ((px - ax) * dx + (py - ay) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    projx = ax + t * dx
    projy = ay + t * dy
    return math.sqrt((px - projx) ** 2 + (py - projy) ** 2)

def _check_non_overlap(inner, eps=0.0):
    """Check that all pairs of inner hexagons are non-overlapping (distance >= eps)."""
    for i in range(len(inner)):
        for j in range(i + 1, len(inner)):
            d = _hex_distance(inner[i], inner[j])
            if d < eps:
                return False
    return True

def _outer_side_needed(inner, outer_center, outer_rot):
    """
    Given inner hexagons and outer center/rotation, find minimum outer side length
    such that all inner hexagons fit inside.
    """
    cx, cy = outer_center
    max_dist = 0.0
    for (ix, iy, iang) in inner:
        # For each vertex of each inner hexagon, compute how far it is from outer center
        # in the direction of the outer hexagon's normals
        verts = _hex_vertices(ix, iy, 1.0, iang)
        for (vx, vy) in verts:
            # Transform to outer hexagon local frame
            dx = vx - cx
            dy = vy - cy
            rot = math.radians(-outer_rot)
            lx = dx * math.cos(rot) - dy * math.sin(rot)
            ly = dx * math.sin(rot) + dy * math.cos(rot)
            
            # For a regular hexagon with side S, a point (lx, ly) is inside if
            # max over i of (lx*cos(i*pi/3) + ly*sin(i*pi/3)) <= S*sqrt(3)/2
            for i in range(6):
                angle = i * math.pi / 3.0
                proj = lx * math.cos(angle) + ly * math.sin(angle)
                # Required side: proj / (sqrt(3)/2) = 2*proj/sqrt(3)
                if proj > 0:
                    needed = 2.0 * proj / math.sqrt(3)
                    max_dist = max(max_dist, needed)
    return max_dist

def _evaluate(inner, outer_center, outer_rot):
    """Return the outer side length needed, or inf if infeasible."""
    if not _check_non_overlap(inner, eps=-0.01):  # allow tiny overlap tolerance
        return float('inf')
    return _outer_side_needed(inner, outer_center, outer_rot)

def construct():
    start_time = time.time()
    
    # Initial guess: hexagonal lattice arrangement
    # 12 hexagons in a 3x4 or 2x6 grid pattern
    # Let's try a few initial configurations
    
    best_score = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_rot = 0.0
    
    # Try several initial configurations
    configs = []
    
    # Config 1: 3 rows of 4
    for row in range(3):
        for col in range(4):
            x = col * 2.0 + (0.0 if row % 2 == 0 else 1.0)
            y = row * math.sqrt(3)
            configs.append([x, y, 0.0])
    
    # Config 2: 2 rows of 6
    configs2 = []
    for row in range(2):
        for col in range(6):
            x = col * 2.0
            y = row * math.sqrt(3)
            configs2.append([x, y, 0.0])
    
    # Config 3: hexagonal cluster
    configs3 = []
    # Center + 5 around + 6 more
    center = [0.0, 0.0, 0.0]
    configs3.append(center)
    for i in range(6):
        angle = i * math.pi / 3
        configs3.append([2.0 * math.cos(angle), 2.0 * math.sin(angle), 0.0])
    for i in range(6):
        angle = i * math.pi / 3 + math.pi / 6
        configs3.append([2.0 * math.sqrt(3) * math.cos(angle), 2.0 * math.sqrt(3) * math.sin(angle), 0.0])
    
    all_configs = [configs, configs2, configs3]
    
    for cfg in all_configs:
        if time.time() > DEADLINE:
            break
        # Center the configuration
        xs = [c[0] for c in cfg]
        ys = [c[1] for c in cfg]
        cx = (max(xs) + min(xs)) / 2.0
        cy = (max(ys) + min(ys)) / 2.0
        centered = [[x - cx, y - cy, a] for x, y, a in cfg]
        
        for rot in [0.0, 30.0, 60.0, 90.0, 120.0, 150.0]:
            if time.time() > DEADLINE:
                break
            score = _evaluate(centered, (0.0, 0.0), rot)
            if score < best_score:
                best_score = score
                best_inner = [list(c) for c in centered]
                best_center = (0.0, 0.0)
                best_rot = rot
    
    # Local optimization using gradient-free method
    if time.time() < DEADLINE - 10 and best_inner is not None:
        from scipy.optimize import minimize
        
        def objective(params):
            if time.time() > DEADLINE:
                return 1e10
            # params: for each hexagon: x, y, angle (3*N params) + outer center x, y (2) + outer rot (1)
            inner_params = params[:3 * N]
            outer_cx = params[3 * N]
            outer_cy = params[3 * N + 1]
            outer_rot = params[3 * N + 2]
            
            inner = []
            for i in range(N):
                x = inner_params[3 * i]
                y = inner_params[3 * i + 1]
                a = inner_params[3 * i + 2]
                inner.append([x, y, a])
            
            # Check non-overlap
            for i in range(N):
                for j in range(i + 1, N):
                    d = _hex_distance(inner[i], inner[j])
                    if d < 0.0:
                        return 1e10 + abs(d) * 1000
            
            return _outer_side_needed(inner, (outer_cx, outer_cy), outer_rot)
        
        # Initial parameters
        init_params = []
        for c in best_inner:
            init_params.extend([c[0], c[1], c[2]])
        init_params.extend([best_center[0], best_center[1], best_rot])
        
        # Try a few optimization attempts
        for attempt in range(3):
            if time.time() > DEADLINE - 5:
                break
            
            if attempt == 0:
                x0 = init_params
            elif attempt == 1:
                # Perturb
                x0 = [p + 0.1 * (2 * (i % 3 == 0) - 1) * (i % 2) for i, p in enumerate(init_params)]
            else:
                x0 = [p + 0.5 * ((i * 7) % 5 - 2) / 5 for i, p in enumerate(init_params)]
            
            try:
                result = minimize(
                    objective, x0, method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6}
                )
                if result.fun < best_score and result.fun < 1e9:
                    best_score = result.fun
                    inner_params = result.x[:3 * N]
                    best_inner = []
                    for i in range(N):
                        best_inner.append([inner_params[3 * i], inner_params[3 * i + 1], inner_params[3 * i + 2]])
                    best_center = (result.x[3 * N], result.x[3 * N + 1])
                    best_rot = result.x[3 * N + 2]
            except Exception:
                pass
    
    # If we still have a valid solution, return it
    if best_inner is not None and best_score < 1e9:
        return best_inner, best_center, best_score, best_rot
    
    # Fallback: return a valid but not optimal solution
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0
