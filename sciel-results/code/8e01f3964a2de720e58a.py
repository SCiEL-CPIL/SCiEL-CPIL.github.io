"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

N = 12

def _hex_vertices(cx, cy, side, rot_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy), side length 'side',
    rotated by rot_deg degrees. Flat-top orientation when rot=0."""
    verts = []
    for k in range(6):
        angle = math.radians(60.0 * k + rot_deg)
        x = cx + side * math.cos(angle)
        y = cy + side * math.sin(angle)
        verts.append((x, y))
    return verts

def _point_in_hex(px, py, cx, cy, side, rot_deg):
    """Check if point (px,py) is inside or on boundary of hexagon."""
    # Transform to hexagon's local frame
    dx = px - cx
    dy = py - cy
    rot = math.radians(-rot_deg)
    lx = dx * math.cos(rot) - dy * math.sin(rot)
    ly = dx * math.sin(rot) + dy * math.cos(rot)
    
    # For a flat-top hexagon with side s, the vertices are at angles 0,60,...,300
    # The hexagon can be described by: max over 3 axes of |projection| <= s
    # Axis 1: horizontal (angle 0): |lx| <= s
    # Axis 2: 60 deg: |lx*cos60 + ly*sin60| <= s
    # Axis 3: 120 deg: |lx*cos120 + ly*sin120| <= s
    c60 = math.cos(math.radians(60))
    s60 = math.sin(math.radians(60))
    
    a1 = abs(lx)
    a2 = abs(lx * c60 + ly * s60)
    a3 = abs(lx * (-c60) + ly * s60)
    
    return a1 <= side + 1e-9 and a2 <= side + 1e-9 and a3 <= side + 1e-9

def _hexagon_in_hexagon(inner_verts, outer_verts):
    """Check if all vertices of inner hexagon are inside outer hexagon.
    For convex polygons, this is sufficient."""
    for (px, py) in inner_verts:
        # Check if point is inside outer hexagon
        # Use cross product method for convex polygon
        n = len(outer_verts)
        inside = True
        for i in range(n):
            x1, y1 = outer_verts[i]
            x2, y2 = outer_verts[(i + 1) % n]
            # Cross product of (x2-x1, y2-y1) and (px-x1, py-y1)
            cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
            if cross < -1e-9:
                inside = False
                break
        if not inside:
            return False
    return True

def _min_side_for_config(inner_positions, outer_center=(0.0, 0.0), outer_rot=0.0):
    """Find the minimum side length of outer hexagon that contains all inner hexagons.
    Uses binary search on side length."""
    lo = 0.0
    hi = 100.0
    
    for _ in range(50):
        mid = (lo + hi) / 2.0
        outer_verts = _hex_vertices(outer_center[0], outer_center[1], mid, outer_rot)
        
        all_inside = True
        for (cx, cy, rot) in inner_positions:
            inner_verts = _hex_vertices(cx, cy, 1.0, rot)
            if not _hexagon_in_hexagon(inner_verts, outer_verts):
                all_inside = False
                break
        
        if all_inside:
            hi = mid
        else:
            lo = mid
    
    return hi

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Strategy: Try several symmetric arrangements of 12 unit hexagons
    # and find the minimum enclosing regular hexagon side length.
    
    # Configuration 1: 3 rows of 4, offset like honeycomb
    # Honeycomb packing: rows offset by sqrt(3)/2 in y, 1.5 in x
    # For hexagons of side 1, distance between centers in honeycomb:
    # Horizontal neighbors: 2*sqrt(3) apart? No, let me think again.
    
    # Regular hexagon side 1: 
    # Width (flat-top) = 2*side = 2
    # Height = sqrt(3)*side = sqrt(3)
    # In honeycomb, adjacent hexagons center-to-center distance = 2*sqrt(3)/2 * ... 
    # Actually for flat-top hexagons packed in honeycomb:
    # Row spacing (vertical) = 1.5 * side? No.
    # For flat-top hexagons with side s:
    #   width = 2s, height = sqrt(3)*s
    #   Horizontal spacing between centers in same row: 3s/2 * 2? 
    #   Let me just use known honeycomb distances.
    
    # For regular hexagons of side 1 packed in a honeycomb:
    # The distance between centers of adjacent hexagons is 2*sqrt(3)/2 = sqrt(3)? 
    # No. Let me compute: two flat-top hexagons side by side horizontally.
    # Each has width 2. They share an edge. Center-to-center = 2*sqrt(3)/2 * 2?
    # 
    # A flat-top hexagon with side s has vertices at:
    # (s,0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s,0), (-s/2,-s*sqrt(3)/2), (s/2,-s*sqrt(3)/2)
    # Width = 2s, height = s*sqrt(3)
    # 
    # Two such hexagons sharing a vertical edge: centers separated by 2s horizontally? No.
    # They share an edge of length s. The shared edge is vertical, from (s/2, -s*sqrt(3)/2) to (s/2, s*sqrt(3)/2) for the left hex, which is (-s/2, ...) for the right hex.
    # So center-to-center = s - (-s/2) + ... wait. Left hex right edge at x=s/2, right hex left edge at x=s/2 (if centered at s). 
    # So centers at 0 and s? No: left hex centered at 0, rightmost vertex at (s,0) and (s/2, ±s*sqrt(3)/2). The right edge is the line segment from (s/2, s*sqrt(3)/2) to (s/2, -s*sqrt(3)/2)? No, that's not right either.
    
    # Let me just place them and check.
    
    configs = []
    
    # Config A: 4x3 grid with honeycomb offset
    # Row 0: 4 hexagons at y=0
    # Row 1: 4 hexagons at y=sqrt(3), x offset by 1.5
    # Row 2: 4 hexagons at y=2*sqrt(3)
    # For flat-top hexagons side 1:
    # Horizontal center spacing in same row: 3 (since width is 2, but they interlock... actually for flat-top, horizontal neighbors in honeycomb are 3 apart? Let's just try 3.0)
    # Vertical spacing: sqrt(3)
    # X offset for odd rows: 1.5
    
    s3 = math.sqrt(3)
    
    # Honeycomb arrangement, flat-top
    # Actually, for flat-top hexagons, the natural honeycomb has:
    # - Horizontal spacing: 3 (centers)
    # - Vertical spacing: s3
    # - Odd rows shifted by 1.5
    
    row0 = [(1.5 * i, 0.0, 0.0) for i in range(4)]
    row1 = [(1.5 * i + 0.75, s3 * 0.5, 0.0) for i in range(4)]
    row2 = [(1.5 * i, s3, 0.0) for i in range(4)]
    
    # Wait, that's not right for honeycomb. Let me reconsider.
    # In a honeycomb of flat-top hexagons:
    # The six neighbors of a hexagon at (0,0) are at:
    # (3,0), (-3,0), (1.5, s3/2), (1.5, -s3/2), (-1.5, s3/2), (-1.5, -s3/2)
    # So horizontal spacing is 3, diagonal spacing is sqrt(1.5^2 + (s3/2)^2) = sqrt(2.25 + 0.75) = sqrt(3)
    
    # So for a 4x3 honeycomb:
    # Row 0 (y=0): x = 0, 3, 6, 9
    # Row 1 (y=s3/2): x = 1.5, 4.5, 7.5, 10.5
    # Row 2 (y=s3): x = 0, 3, 6, 9
    
    config_a = []
    for i in range(4):
        config_a.append([3.0 * i, 0.0, 0.0])
    for i in range(4):
        config_a.append([3.0 * i + 1.5, s3 / 2.0, 0.0])
    for i in range(4):
        config_a.append([3.0 * i, s3, 0.0])
    configs.append(config_a)
    
    # Config B: 3x4 honeycomb (transposed)
    config_b = []
    for i in range(3):
        for j in range(4):
            x = 3.0 * j
            y = s3 * i
            if i % 2 == 1:
                x += 1.5
            config_b.append([x, y, 0.0])
    configs.append(config_b)
    
    # Config C: 6x2 grid
    config_c = []
    for i in range(6):
        for j in range(2):
            x = 3.0 * i
            y = s3 * j
            if j % 2 == 1:
                x += 1.5
            config_c.append([x, y, 0.0])
    configs.append(config_c)
    
    # Config D: 2x6 grid
    config_d = []
    for i in range(2):
        for j in range(6):
            x = 3.0 * j
            y = s3 * i
            if i % 2 == 1:
                x += 1.5
            config_d.append([x, y, 0.0])
    configs.append(config_d)
    
    best_side = float('inf')
    best_config = None
    best_outer_center = (0.0, 0.0)
    best_outer_rot = 0.0
    
    for config in configs:
        if time.time() > deadline - 10.0:
            break
        
        # Center the configuration
        xs = [p[0] for p in config]
        ys = [p[1] for p in config]
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        
        centered = [[p[0] - cx, p[1] - cy, p[2]] for p in config]
        
        # Try different outer rotations
        for rot in [0.0, 30.0]:
            if time.time() > deadline - 10.0:
                break
            
            side = _min_side_for_config(centered, (0.0, 0.0), rot)
            if side < best_side:
                best_side = side
                best_config = centered
                best_outer_rot = rot
    
    # If we found a good config, return it
    if best_config is not None and best_side < float('inf'):
        inner = [[p[0], p[1], p[2]] for p in best_config]
        return inner, (0.0, 0.0), best_side, best_outer_rot
    
    # Fallback: return something valid
    inner = [[3.0 * i, 0.0, 0.0] for i in range(N)]
    return inner, (0.0, 0.0), 50.0, 0.0
# EVOLVE-BLOCK-END
