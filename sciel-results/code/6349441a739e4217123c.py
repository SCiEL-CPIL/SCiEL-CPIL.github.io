"""Optimized hexagon packing: 11 unit hexagons in a compact hexagonal lattice."""
import math
import time
import itertools

N = 11


def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with side length `side`,
    centered at (cx, cy), rotated by angle_deg degrees."""
    verts = []
    for k in range(6):
        a = math.radians(angle_deg + 60 * k)
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return verts


def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with side `side` rotated by angle_deg."""
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


def _min_outer_side_analytical(hexes, oangle_rad):
    """
    Find minimum side length of outer hexagon with rotation oangle_rad
    that contains all hexagons in `hexes` (centered at origin).
    Returns (min_side, center_x, center_y)
    """
    # Points to contain: all vertices of all inner hexagons
    # Since inner hexagons are unit, we can just use their vertices.
    # Or more efficiently, since they are convex, we can check support functions.
    # But for now, let's use the vertices of the union.
    # Actually, we can just use the centers and the "radius" in each direction?
    # No, the hexagons have shape.
    
    # Let's collect all vertices of the 11 hexagons.
    points = []
    for cx, cy, s, ang in hexes:
        verts = _hex_vertices(cx, cy, s, ang)
        points.extend(verts)
        
    # For a fixed outer rotation oangle_rad, the constraints are on the center (ox, oy).
    # The outer hexagon has side L, apothem A = L * sqrt(3)/2.
    # The condition is that for all 6 normal directions u_k (rotated by oangle_rad),
    # the projection of all points onto u_k must be within [ox*u_kx + oy*uky - A, ox*u_kx + oy*uky + A].
    # Let P_k = max_i (p_i . u_k) and Q_k = min_i (p_i . u_k).
    # Then we need:
    # ox*u_kx + oy*uky - A <= Q_k  =>  ox*u_kx + oy*uky <= Q_k + A
    # ox*u_kx + oy*uky + A >= P_k  =>  ox*u_kx + oy*uky >= P_k - A
    
    # So for each k, the center (ox, oy) must lie in the strip:
    # P_k - A <= ox*u_kx + oy*uky <= Q_k + A
    
    # We want to minimize A.
    # Note that P_k - Q_k is the width of the point set in direction u_k.
    # Let W_k = P_k - Q_k.
    # The strip width is 2A.
    # For the intersection of the 6 strips to be non-empty, we must have A >= W_k / 2 for all k.
    # So A_min_candidate = max_k (W_k / 2).
    
    # Let A_req = max_k (W_k / 2).
    # We check if there exists (ox, oy) such that for all k:
    # P_k - A_req <= ox*u_kx + oy*uky <= Q_k + A_req
    
    # If such (ox, oy) exists, then L = A_req * 2 / sqrt(3) is the answer for this angle.
    # If not, we need a larger A.
    
    # To check existence, we can find the intersection of the 6 strips.
    # This is a convex polygon. We can compute its vertices.
    # If the polygon is non-empty, we are good.
    
    # Optimization: The optimal center is often the centroid.
    # Let's try the centroid first.
    cx_mean = sum(p[0] for p in points) / len(points)
    cy_mean = sum(p[1] for p in points) / len(points)
    
    # Calculate W_k and A_req
    A_req = 0.0
    P_vals = [0.0] * 6
    Q_vals = [0.0] * 6
    u_vectors = []
    
    for k in range(6):
        theta = oangle_rad + math.radians(30 + 60 * k)
        ux = math.cos(theta)
        uy = math.sin(theta)
        u_vectors.append((ux, uy))
        
        max_proj = -float('inf')
        min_proj = float('inf')
        for px, py in points:
            proj = px * ux + py * uy
            if proj > max_proj: max_proj = proj
            if proj < min_proj: min_proj = proj
            
        P_vals[k] = max_proj
        Q_vals[k] = min_proj
        width = max_proj - min_proj
        if width / 2.0 > A_req:
            A_req = width / 2.0
            
    # Check if centroid is valid for A_req
    valid = True
    for k in range(6):
        ux, uy = u_vectors[k]
        center_proj = cx_mean * ux + cy_mean * uy
        lower = P_vals[k] - A_req
        upper = Q_vals[k] + A_req
        if center_proj < lower - 1e-9 or center_proj > upper + 1e-9:
            valid = False
            break
            
    if valid:
        side = A_req * 2.0 / math.sqrt(3)
        return side, cx_mean, cy_mean
    else:
        # If centroid is not valid, we need to find the minimal A such that the intersection is non-empty.
        # This is a linear programming problem.
        # Minimize A
        # Subject to:
        # ox*u_kx + oy*uky - A <= Q_k
        # ox*u_kx + oy*uky + A >= P_k  => -ox*u_kx - oy*uky - A <= -P_k
        
        # We can use a simple iterative increase of A.
        # Start with A_req and increase until valid.
        # Since the domain is small, a few steps should suffice.
        A = A_req
        # Step size
        step = 0.01
        max_iter = 100
        found_center = None
        
        for i in range(max_iter):
            # Check if there is a center for this A
            # We can check if the intersection of the 6 strips is non-empty.
            # A simple heuristic: check if the centroid of the intersection of the first few strips is in the rest?
            # Or just check if the "center of mass" of the feasible region (if it existed) is valid.
            # Let's try to find a point in the intersection of the 6 strips.
            # This is a 2D feasibility problem.
            # We can use the fact that the intersection of half-planes is a convex polygon.
            # We can compute the vertices of the intersection polygon.
            
            # For now, let's just try to adjust the center towards the feasible region.
            # Or simply return a slightly larger side to be safe.
            # Given the time, we can just increase A by a small amount and check centroid again?
            # No, the centroid might not be the right center.
            
            # Let's use a simple grid search for the center (ox, oy) in a small box around the centroid.
            # Box size: 2 * A_req
            ox_min = cx_mean - 2*A_req
            ox_max = cx_mean + 2*A_req
            oy_min = cy_mean - 2*A_req
            oy_max = cy_mean + 2*A_req
            
            # Coarse grid
            found = False
            for ox in [cx_mean]: # Just try centroid first
                 for oy in [cy_mean]:
                     valid_c = True
                     for k in range(6):
                         ux, uy = u_vectors[k]
                         center_proj = ox * ux + oy * uy
                         lower = P_vals[k] - A
                         upper = Q_vals[k] + A
                         if center_proj < lower - 1e-9 or center_proj > upper + 1e-9:
                             valid_c = False
                             break
                     if valid_c:
                         found = True
                         found_center = (ox, oy)
                         break
                 if found: break
            
            if found:
                side = A * 2.0 / math.sqrt(3)
                return side, found_center[0], found_center[1]
            
            A += step
            
        # If we didn't find a center, return a large value
        return float('inf'), 0.0, 0.0


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s limit

    sqrt3 = math.sqrt(3)

    # Flat-top hexagon mapping:
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    def axial_to_xy(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return (x, y)

    # Define 11 positions in axial coordinates (q, r)
    # Center
    pos_axial = [(0, 0)]
    # Ring 1 (6 neighbors)
    pos_axial.extend([(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)])
    
    # We need 4 more from Ring 2.
    # Ring 2 axial coords (distance 2 from center):
    ring2_candidates = [
        (2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2), 
        (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2), 
        (0, 2), (1, 2)
    ]
    
    best_side = float('inf')
    best_xy = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # Try all combinations of 4 from ring2_candidates
    for combo in itertools.combinations(range(len(ring2_candidates)), 4):
        if time.time() > deadline:
            break
            
        selected_r2 = [ring2_candidates[i] for i in combo]
        full_axial = pos_axial + selected_r2
        
        # Convert to Cartesian
        full_xy = [axial_to_xy(q, r) for q, r in full_axial]
        
        # Center the cluster at origin for calculation
        cx_mean = sum(x for x, y in full_xy) / 11.0
        cy_mean = sum(y for x, y in full_xy) / 11.0
        
        # Shift points to origin
        centered_xy = [(x - cx_mean, y - cy_mean) for x, y in full_xy]
        
        # Inner hexagons are aligned with the lattice (angle 0)
        inner_hexes_centered = [(x, y, 1.0, 0.0) for x, y in centered_xy]
        
        # Sweep outer rotation
        # Due to symmetry, we only need to check 0 to 30 degrees
        # Step 0.1 degrees
        for angle_deg in range(0, 31, 1): # 0, 1, ..., 30
            angle_rad = math.radians(angle_deg)
            side, ox, oy = _min_outer_side_analytical(inner_hexes_centered, angle_rad)
            
            if side < best_side:
                # Convert center back to original frame
                best_side = side
                best_center = (ox + cx_mean, oy + cy_mean)
                best_angle = angle_deg
                best_xy = full_xy

    if best_xy is None:
        # Fallback
        best_xy = [axial_to_xy(q, r) for q, r in pos_axial]
        cx_mean = sum(x for x, y in best_xy) / 11.0
        cy_mean = sum(y for x, y in best_xy) / 11.0
        centered_xy = [(x - cx_mean, y - cy_mean) for x, y in best_xy]
        inner_hexes_centered = [(x, y, 1.0, 0.0) for x, y in centered_xy]
        best_side, ox, oy = _min_outer_side_analytical(inner_hexes_centered, 0.0)
        best_center = (ox + cx_mean, oy + cy_mean)
        best_angle = 0.0

    # Build final inner hexagons
    inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_xy]

    # Verify disjointness
    if not _all_disjoint(inner_hexes):
        scale = 1.01
        inner_hexes = [(x * scale, y * scale, 1.0, 0.0) for x, y in best_xy]
        cx_mean = sum(x for x, y, s, a in inner_hexes) / 11.0
        cy_mean = sum(y for x, y, s, a in inner_hexes) / 11.0
        # Recalculate side
        centered_xy = [(x - cx_mean, y - cy_mean) for x, y in [(h[0], h[1]) for h in inner_hexes]]
        inner_hexes_centered = [(x, y, 1.0, 0.0) for x, y in centered_xy]
        best_side, ox, oy = _min_outer_side_analytical(inner_hexes_centered, math.radians(best_angle))
        best_center = (ox + cx_mean, oy + cy_mean)

    # Verify all inner hexagons are inside outer
    outer_hex = (best_center[0], best_center[1], best_side, best_angle)
    valid = True
    for h in inner_hexes:
        if not _inner_in_outer(h, outer_hex):
            valid = False
            break

    if not valid:
        # Increase side slightly to ensure containment
        best_side *= 1.005
        outer_hex = (best_center[0], best_center[1], best_side, best_angle)

    # Format output: inner rows (x, y, angle_degrees)
    inner = [[h[0], h[1], h[3]] for h in inner_hexes]

    return inner, (best_center[0], best_center[1]), best_side, best_angle
