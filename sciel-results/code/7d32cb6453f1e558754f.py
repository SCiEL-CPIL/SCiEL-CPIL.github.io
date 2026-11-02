"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

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

def _hex_support_function(cx, cy, angle_deg, nx, ny):
    """Compute the support function of a unit hexagon at (cx, cy) with rotation angle_deg
    in direction (nx, ny). Returns max dot product over all vertices."""
    verts = _get_hex_vertices(cx, cy, angle_deg)
    return max((vx * nx + vy * ny) for vx, vy in verts)

def _hex_clearance_between(h1, h2):
    """Compute the minimum clearance (distance) between two unit hexagons.
    h1 = (x1, y1, ang1), h2 = (x2, y2, ang2)
    Returns positive value if separated, negative if overlapping."""
    # For two convex polygons, the minimum distance can be found by checking
    # all pairs of edges for closest approach, but for hexagons we can use
    # the support function approach.
    # 
    # Two hexagons are disjoint iff for every direction, the sum of support
    # functions in opposite directions is less than the distance between centers
    # projected in that direction.
    #
    # Simpler: check if any vertex of one is inside the other, and if any
    # edges intersect. But for efficiency, use the separating axis theorem
    # with the 12 edge normals (6 from each hexagon).
    
    x1, y1, a1 = h1
    x2, y2, a2 = h2
    
    # Get all edge normals from both hexagons
    normals = []
    for ang in [a1, a2]:
        for k in range(6):
            theta = math.radians(ang + 30 + 60 * k)
            normals.append((math.cos(theta), math.sin(theta)))
    
    min_clearance = float('inf')
    
    for nx, ny in normals:
        # Project both hexagons onto this axis
        sup1 = _hex_support_function(x1, y1, a1, nx, ny)
        sup1_neg = _hex_support_function(x1, y1, a1, -nx, -ny)
        sup2 = _hex_support_function(x2, y2, a2, nx, ny)
        sup2_neg = _hex_support_function(x2, y2, a2, -nx, -ny)
        
        # The gap along this axis: if sup1_neg < sup2 and sup2_neg < sup1, they overlap
        # The clearance along this axis is min(sup2 - sup1_neg, sup1 - sup2_neg) if positive
        gap1 = sup2 - sup1_neg  # distance from far side of h1 to near side of h2
        gap2 = sup1 - sup2_neg  # distance from far side of h2 to near side of h1
        
        # If both gaps are negative, there's overlap along this axis
        # The minimum clearance is the minimum over all axes of the maximum gap
        # Actually, for disjoint convex bodies, the minimum distance is the maximum
        # over all separating axes of the gap.
        
        # The clearance in this direction is the minimum of the two gaps
        # (how far apart they are when projected)
        clearance_axis = min(gap1, gap2)
        
        if clearance_axis < min_clearance:
            min_clearance = clearance_axis
    
    return min_clearance

def _check_disjoint(inner_hexes, min_gap=1e-6):
    """Check that all pairs of inner hexagons are disjoint with at least min_gap separation."""
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            clearance = _hex_clearance_between(inner_hexes[i], inner_hexes[j])
            if clearance < min_gap:
                return False
    return True

def _min_enclosing_hex_side_lp(inner_hexes, angle_deg):
    """For a fixed outer hexagon angle, find optimal center and side length using LP."""
    # Collect all vertices
    all_vertices = []
    for ix, iy, iangle in inner_hexes:
        verts = _get_hex_vertices(ix, iy, iangle)
        all_vertices.extend(verts)
    
    n_pts = len(all_vertices)
    if n_pts == 0:
        return 0.0, 0.0, 0.0, angle_deg
    
    # LP: minimize r
    # Variables: [cx, cy, r]
    # Constraints: for each vertex (vx, vy) and each direction k:
    #   (vx - cx) * nx_k + (vy - cy) * ny_k <= r
    #   => -nx_k * cx - ny_k * cy - r <= -vx*nx_k - vy*ny_k
    # Also: r >= 0
    
    A_ub = []
    b_ub = []
    
    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        for vx, vy in all_vertices:
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-vx * nx - vy * ny)
    
    # r >= 0  =>  -r <= 0
    A_ub.append([0.0, 0.0, -1.0])
    b_ub.append(0.0)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    c = np.array([0.0, 0.0, 1.0])  # minimize r
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
    
    if result.success:
        cx_opt = result.x[0]
        cy_opt = result.x[1]
        r_opt = result.x[2]
    else:
        # Fallback: use centroid and compute required r
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
        cx_opt = cx_mean
        cy_opt = cy_mean
    
    side = r_opt / (math.sqrt(3) / 2)
    return cx_opt, cy_opt, side, angle_deg

def _min_enclosing_hex_side(inner_hexes):
    """Find minimum side length of a regular hexagon that contains all inner hexagons.
    Returns (center_x, center_y, side_length, angle_deg)."""
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # First pass: coarse sweep
    angles_coarse = np.arange(0, 60, 2.0)
    for angle_deg in angles_coarse:
        cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg))
        if side < best_side:
            best_side = side
            best_center = (cx, cy)
            best_angle = float(angle_deg)
    
    # Second pass: fine sweep around the best angle
    if best_side < float('inf'):
        angles_fine = np.arange(best_angle - 5, best_angle + 5, 0.5)
        for angle_deg in angles_fine:
            if angle_deg < 0 or angle_deg >= 60:
                continue
            cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg))
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)
    
    # Third pass: even finer sweep
    if best_side < float('inf'):
        angles_finer = np.arange(best_angle - 1, best_angle + 1, 0.1)
        for angle_deg in angles_finer:
            if angle_deg < 0 or angle_deg >= 60:
                continue
            cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg))
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)
    
    return best_center[0], best_center[1], best_side, best_angle

def _greedy_max_min_clearance_placement():
    """Place hexagons one by one, each maximizing its minimum clearance to already-placed
    hexagons, using a local optimization for each placement."""
    
    # Start with the first hexagon at origin
    placed = [[0.0, 0.0, 30.0]]
    
    for step in range(1, N):
        if time.time() > _DEADLINE - 15.0:
            break
        
        # Try multiple starting positions and angles
        best_placement = None
        best_clearance = -float('inf')
        
        # Candidate starting positions: on a grid around the current cluster
        # The cluster radius grows as we add hexagons
        cluster_radius = step * 1.8  # approximate
        
        # Try positions in a ring around the cluster
        num_candidates = 24
        for trial in range(num_candidates):
            if time.time() > _DEADLINE - 15.0:
                break
            
            # Spread candidates around the circle
            theta = 2 * math.pi * trial / num_candidates + 0.1 * step
            r = cluster_radius * 0.8 + 0.5
            
            x0 = r * math.cos(theta)
            y0 = r * math.sin(theta)
            
            # Try a few angles
            for ang0 in [30.0, 0.0, 60.0]:
                # Local optimization: maximize min clearance to placed hexagons
                def objective(params):
                    x, y, ang = params[0], params[1], math.degrees(params[2])
                    candidate = [x, y, ang]
                    
                    # Compute minimum clearance to all placed hexagons
                    min_clr = float('inf')
                    for ph in placed:
                        clr = _hex_clearance_between(candidate, ph)
                        if clr < min_clr:
                            min_clr = clr
                    
                    # We want to maximize min_clr, so minimize -min_clr
                    return -min_clr
                
                x0_rad = x0
                y0_rad = y0
                ang0_rad = math.radians(ang0)
                
                try:
                    result = minimize(objective, [x0_rad, y0_rad, ang0_rad],
                                     method='Nelder-Mead',
                                     options={'maxiter': 100, 'xatol': 0.01, 'fatol': 0.001})
                    
                    if result.success:
                        x_opt = result.x[0]
                        y_opt = result.x[1]
                        ang_opt = math.degrees(result.x[2])
                        clr = -result.fun
                        
                        if clr > best_clearance:
                            best_clearance = clr
                            best_placement = [x_opt, y_opt, ang_opt]
                except Exception:
                    pass
        
        if best_placement is not None:
            placed.append(best_placement)
        else:
            # Fallback: place at a reasonable position
            theta = 2 * math.pi * step / 6
            r = cluster_radius
            placed.append([r * math.cos(theta), r * math.sin(theta), 30.0])
    
    return placed

def _refine_placement(inner_hexes):
    """Refine the placement using local optimization to maximize minimum pairwise clearance."""
    if len(inner_hexes) < 2:
        return inner_hexes
    
    # Flatten to parameters: x1, y1, a1, x2, y2, a2, ...
    def flatten(hexes):
        params = []
        for x, y, a in hexes:
            params.extend([x, y, math.radians(a)])
        return params
    
    def unflatten(params):
        hexes = []
        for i in range(len(params) // 3):
            hexes.append([params[3*i], params[3*i+1], math.degrees(params[3*i+2])])
        return hexes
    
    def objective(params):
        hexes = unflatten(params)
        n = len(hexes)
        min_clr = float('inf')
        for i in range(n):
            for j in range(i + 1, n):
                clr = _hex_clearance_between(hexes[i], hexes[j])
                if clr < min_clr:
                    min_clr = clr
        return -min_clr
    
    x0 = flatten(inner_hexes)
    
    try:
        result = minimize(objective, x0, method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 0.005, 'fatol': 0.0005})
        
        if result.success:
            refined = unflatten(result.x)
            # Verify disjointness
            if _check_disjoint(refined, min_gap=1e-5):
                return refined
    except Exception:
        pass
    
    return inner_hexes

def construct():
    global _START_TIME, _DEADLINE
    
    # Check deadline
    if time.time() > _DEADLINE - 5.0:
        # Return a safe fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Strategy: Use greedy max-min clearance placement, then refine,
    # then compute optimal enclosing hexagon.
    
    # Initial placement using greedy approach
    inner = _greedy_max_min_clearance_placement()
    
    # Ensure we have exactly N hexagons
    while len(inner) < N:
        # Add more hexagons at reasonable positions
        idx = len(inner)
        theta = 2 * math.pi * idx / 6
        r = idx * 1.5
        inner.append([r * math.cos(theta), r * math.sin(theta), 30.0])
    
    # Refine placement to maximize minimum clearance
    if time.time() < _DEADLINE - 20.0:
        inner = _refine_placement(inner)
    
    # Verify disjointness
    if not _check_disjoint(inner, min_gap=1e-5):
        # Fallback to lattice placement
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
    
    # Compute minimum enclosing hexagon
    if time.time() > _DEADLINE - 10.0:
        # Quick estimate
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
    
    cx, cy, side, angle = _min_enclosing_hex_side(inner)
    
    return inner, (cx, cy), side, angle
# EVOLVE-BLOCK-END
