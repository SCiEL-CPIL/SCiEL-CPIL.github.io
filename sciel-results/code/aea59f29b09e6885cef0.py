"""
Pack unit regular hexagons inside a regular hexagon, minimizing outer side length.
Uses a grid-based placement with local optimization (simulated annealing / hill climbing).
"""
import math
import random
import time

N = 11

# Time budget guard
_START_TIME = None

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _hex_vertex(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy), side length, rotated by angle_deg."""
    rad = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = rad + math.radians(60 * i)
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return verts

def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with given side and angle."""
    # Transform to hex local frame
    rad = math.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    # Rotate back
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    lx = cos_a * dx + sin_a * dy
    ly = -sin_a * dx + cos_a * dy
    
    # In local frame, hexagon has vertices at angles 0,60,...,300 with radius = side
    # Actually for a regular hexagon with side s, the distance from center to vertex is s.
    # The hexagon can be described as: max over 6 directions of dot product <= s/2 * sqrt(3)? 
    # No. Let me think again.
    # A regular hexagon with side length s has apothem (distance from center to edge) = s * sqrt(3)/2.
    # The 6 edges are perpendicular to directions at angles 30, 90, 150, 210, 270, 330 degrees (for a "flat-top" or "pointy-top" depending on convention).
    # Actually, if vertices are at angles 0,60,..., then edges are perpendicular to 30,90,...
    # Let's use the standard: a hexagon centered at origin with vertices at (s*cos(k*60), s*sin(k*60)) for k=0..5.
    # The half-width in the x-direction (to a flat edge) is s*sqrt(3)/2 if the hex is oriented with a vertex at angle 0? 
    # No, if vertex at angle 0, then the rightmost point is at x=s. The edge between vertex at 0 and 60 degrees...
    # Let me just use the cross-product method or check against all 6 edges.
    
    # Better: use the fact that a regular hexagon is the intersection of 6 half-planes.
    # For a hex with vertices at angles theta_k = rad + k*pi/3, the edges are between consecutive vertices.
    # The normal to the edge between vertex k and k+1 points outward.
    # The distance from center to each edge (apothem) is a = s * cos(pi/6) = s * sqrt(3)/2.
    # The outward normal direction for the edge opposite to vertex at angle alpha is at angle alpha + pi (no, that's not right).
    
    # Simpler: For a regular hexagon centered at origin with "pointy top" (vertex at 90 deg) or "flat top"...
    # Let me just compute the 6 vertices in local frame and check if point is inside using winding number or cross products.
    
    # In local frame (unrotated), vertices are at:
    # v_k = (s * cos(k*pi/3), s * sin(k*pi/3)) for k=0..5
    # The point (lx, ly) is inside if for all edges, the point is on the interior side.
    
    s = side
    # Vertices in local frame
    vx = [s * math.cos(k * math.pi / 3) for k in range(6)]
    vy = [s * math.sin(k * math.pi / 3) for k in range(6)]
    
    # For each edge from v_k to v_{k+1}, check that (lx,ly) is on the left side (interior)
    for k in range(6):
        x1, y1 = vx[k], vy[k]
        x2, y2 = vx[(k + 1) % 6], vy[(k + 1) % 6]
        # Edge vector
        ex = x2 - x1
        ey = y2 - y1
        # Vector from v1 to point
        px2 = lx - x1
        py2 = ly - y1
        # Cross product z-component
        cross = ex * py2 - ey * px2
        # For a CCW polygon, interior is to the left, so cross should be >= 0
        if cross < -1e-9:
            return False
    return True

def _hex_overlap(h1, h2):
    """Check if two unit hexagons overlap. h = (cx, cy, angle_deg)."""
    cx1, cy1, a1 = h1
    cx2, cy2, a2 = h2
    
    # Quick bounding box check
    dx = cx1 - cx2
    dy = cy1 - cy2
    if dx * dx + dy * dy > 4.0:  # max distance for non-overlap is 2*side = 2 for unit hex
        return False
    
    # Check if any vertex of one is inside the other
    verts1 = _hex_vertex(cx1, cy1, 1.0, a1)
    verts2 = _hex_vertex(cx2, cy2, 1.0, a2)
    
    for v in verts1:
        if _point_in_hex(v[0], v[1], cx2, cy2, 1.0, a2):
            return True
    for v in verts2:
        if _point_in_hex(v[0], v[1], cx1, cy1, 1.0, a1):
            return True
    
    # Also check edge intersections (approximate by checking more points)
    # For simplicity, also check midpoints of edges
    for i in range(6):
        mx1 = (verts1[i][0] + verts1[(i+1)%6][0]) / 2
        my1 = (verts1[i][1] + verts1[(i+1)%6][1]) / 2
        if _point_in_hex(mx1, my1, cx2, cy2, 1.0, a2):
            return True
        mx2 = (verts2[i][0] + verts2[(i+1)%6][0]) / 2
        my2 = (verts2[i][1] + verts2[(i+1)%6][1]) / 2
        if _point_in_hex(mx2, my2, cx1, cy1, 1.0, a1):
            return True
    
    return False

def _min_side_for_config(positions, angles):
    """Find the minimum side length of outer hexagon (centered at origin, angle 0) that contains all inner hexagons."""
    # The outer hexagon is centered at origin with side S and angle 0.
    # A point (x,y) is inside the outer hex if _point_in_hex(x, y, 0, 0, S, 0)
    # We need the minimum S such that all inner hexagons are fully inside.
    
    # For each inner hex, find the maximum "radius" in the hexagonal norm.
    # The hexagonal norm for a point relative to a hex centered at origin:
    # For a hex with side S and angle 0, the constraint is that for each of the 6 directions,
    # the projection of the point onto the outward normal of each edge must be <= apothem = S * sqrt(3)/2.
    
    # Actually, let's just binary search on S.
    
    lo = 1.0
    hi = 20.0
    
    def fits(S):
        for (cx, cy), ang in zip(positions, angles):
            verts = _hex_vertex(cx, cy, 1.0, ang)
            for (vx, vy) in verts:
                if not _point_in_hex(vx, vy, 0.0, 0.0, S, 0.0):
                    return False
        return True
    
    for _ in range(30):
        mid = (lo + hi) / 2
        if fits(mid):
            hi = mid
        else:
            lo = mid
    
    return hi

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    DEADLINE = 140.0  # Safety margin
    
    # Initial configuration: grid-like arrangement
    # Hexagonal grid spacing for non-overlapping unit hexagons:
    # Horizontal spacing: 2 * side = 2.0 (for flat-top hexes)
    # Actually, for pointy-top hexes with angle 0 (vertex at 0 deg), the horizontal distance
    # between centers of adjacent hexes in the same row is 2*side = 2.0? No.
    # Let me think: two unit hexagons side by side, touching. The distance between centers is 2*apothem = 2*sqrt(3)/2 = sqrt(3) ≈ 1.732? 
    # No, if they share an edge, the distance between centers is 2 * apothem = sqrt(3).
    # But if they're just touching at a vertex, it could be less.
    # For non-overlapping, we need distance >= 2 * side? No, that's for circles.
    # For hexagons, the minimum distance between centers for non-overlap depends on orientation.
    # For two hexes with the same orientation, the minimum center distance for non-overlap is 2 * apothem = sqrt(3) if they share an edge.
    # Actually, the Minkowski sum of two unit hexes... this is getting complex.
    
    # Let's use a simple grid with spacing that guarantees no overlap.
    # For hexagons with angle 0 (vertex pointing right), the horizontal spacing between rows is 2*side = 2.0? 
    # Let me just use a safe spacing: 2.0 in x and sqrt(3) in y for a staggered grid.
    
    n = N
    positions = []
    angles = []
    
    # Hexagonal packing: rows offset
    # For a "pointy-top" hex grid (vertex at top), but our hex has vertex at angle 0 (right)
    # So it's "flat-top" in the sense that the rightmost point is a vertex.
    # Adjacent hexes in a row: separated by 2*side = 2.0 horizontally? 
    # The width of a hex (vertex to vertex) is 2*side = 2.0. So center-to-center in same row: 2.0.
    # Row spacing: the height is 2*apothem = sqrt(3). Staggered rows offset by 1.0 in x.
    
    row_spacing = math.sqrt(3)
    col_spacing = 2.0
    
    num_rows = (n + 1) // 2
    for r in range(num_rows):
        cols_in_row = n - r * 2 if r % 2 == 0 else n - r * 2 + 1
        # This is getting complicated. Let me just place them in a simple grid.
        pass
    
    # Simpler: just place in a linear grid
    # For safety, use spacing 2.2 in both x and y
    spacing = 2.2
    start = -(n // 2) * spacing
    
    for i in range(n):
        x = start + i * spacing
        y = 0.0
        positions.append((x, y))
        angles.append(0.0)
    
    # Optimize using local search
    best_side = _min_side_for_config(positions, angles)
    best_positions = [p[:] for p in positions]
    best_angles = angles[:]
    
    current_positions = [p[:] for p in positions]
    current_angles = angles[:]
    current_side = best_side
    
    temp = 1.0
    cooling = 0.999
    
    step_size = 0.5
    
    while _elapsed() < DEADLINE:
        # Pick a random hex to move
        idx = random.randint(0, n - 1)
        
        old_x, old_y = current_positions[idx]
        old_a = current_angles[idx]
        
        # Random perturbation
        new_x = old_x + random.gauss(0, step_size)
        new_y = old_y + random.gauss(0, step_size)
        new_a = old_a + random.gauss(0, 15.0)
        
        # Try the move
        current_positions[idx] = (new_x, new_y)
        current_angles[idx] = new_a
        
        # Check for overlaps
        has_overlap = False
        for i in range(n):
            for j in range(i + 1, n):
                if _hex_overlap(current_positions[i], current_positions[j]):
                    has_overlap = True
                    break
            if has_overlap:
                break
        
        if has_overlap:
            # Reject
            current_positions[idx] = (old_x, old_y)
            current_angles[idx] = old_a
            temp *= cooling
            continue
        
        # Compute new side
        new_side = _min_side_for_config(current_positions, current_angles)
        
        if new_side < current_side or random.random() < math.exp(-(new_side - current_side) / temp):
            current_side = new_side
            if new_side < best_side:
                best_side = new_side
                best_positions = [p[:] for p in current_positions]
                best_angles = current_angles[:]
        else:
            # Reject
            current_positions[idx] = (old_x, old_y)
            current_angles[idx] = old_a
        
        temp *= cooling
    
    # Return best configuration
    inner = [[best_positions[i][0], best_positions[i][1], best_angles[i]] for i in range(n)]
    return inner, (0.0, 0.0), best_side, 0.0
