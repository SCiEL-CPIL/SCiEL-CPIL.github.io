"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import itertools

N = 12

def hexagon_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
    angle_rad = np.radians(angle_deg)
    angles = np.arange(6) * np.pi / 3 + angle_rad
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def polygon_area(vertices):
    """Shoelace formula for polygon area."""
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px, py) is inside or on boundary of hexagon."""
    # Transform point to local coordinates
    dx = px - cx
    dy = py - cy
    angle_rad = -np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    # For a regular hexagon with side s, the distance from center to edge is s*sqrt(3)/2
    # The hexagon can be described as intersection of 6 half-planes
    # In local coords, edges are at angles 0, 60, 120, 180, 240, 300 degrees
    # Distance from center to each edge along the normal direction is s*sqrt(3)/2
    apothem = side * np.sqrt(3) / 2
    
    # Check all 6 sides
    for i in range(6):
        edge_angle = i * np.pi / 3
        nx = np.cos(edge_angle)
        ny = np.sin(edge_angle)
        dist = lx * nx + ly * ny
        if dist > apothem + 1e-9:
            return False
    return True

def hexagons_disjoint(hex1, hex2):
    """Check if two hexagons are disjoint (no interior overlap)."""
    cx1, cy1, side1, angle1 = hex1
    cx2, cy2, side2, angle2 = hex2
    
    # Quick bounding box check
    r1 = side1
    r2 = side2
    dx = cx1 - cx2
    dy = cy1 - cy2
    dist = np.sqrt(dx*dx + dy*dy)
    if dist > r1 + r2 + 1e-9:
        return True
    
    # Check if any vertex of one is inside the other
    verts1 = hexagon_vertices(cx1, cy1, side1, angle1)
    verts2 = hexagon_vertices(cx2, cy2, side2, angle2)
    
    for v in verts1:
        if point_in_hexagon(v[0], v[1], cx2, cy2, side2, angle2):
            return False
    for v in verts2:
        if point_in_hexagon(v[0], v[1], cx1, cy1, side1, angle1):
            return False
    
    # Check edge intersections
    for i in range(6):
        p1 = verts1[i]
        p2 = verts1[(i+1) % 6]
        for j in range(6):
            p3 = verts2[j]
            p4 = verts2[(j+1) % 6]
            if segments_intersect(p1, p2, p3, p4):
                return False
    
    return True

def segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4."""
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    
    # Check collinear cases
    if d1 == 0 and on_segment(p3, p4, p1): return True
    if d2 == 0 and on_segment(p3, p4, p2): return True
    if d3 == 0 and on_segment(p1, p2, p3): return True
    if d4 == 0 and on_segment(p1, p2, p4): return True
    
    return False

def on_segment(p, q, r):
    """Check if point q lies on segment pr."""
    if min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and \
       min(p[1], r[1]) <= q[1] <= max(p[1], r[1]):
        return True
    return False

def contains_hexagon_in_outer(inner_hex, outer_hex):
    """Check if inner_hex is fully contained in outer_hex."""
    cx_i, cy_i, side_i, angle_i = inner_hex
    cx_o, cy_o, side_o, angle_o = outer_hex
    
    verts_i = hexagon_vertices(cx_i, cy_i, side_i, angle_i)
    for v in verts_i:
        if not point_in_hexagon(v[0], v[1], cx_o, cy_o, side_o, angle_o):
            return False
    return True

def construct():
    """
    Place 12 disjoint unit hexagons inside a regular hexagon, minimizing outer side length.
    Returns: (inner, center, side, angle)
    """
    # Strategy: Use a hexagonal grid arrangement
    # 12 hexagons can be arranged in a hexagonal pattern
    # A hexagon of "radius" 2 has 1 + 6 + 12 = 19 hexagons
    # A hexagon of "radius" 1 has 7 hexagons
    # 12 hexagons: we can use a 3-4-3-3-4-3 pattern or similar
    
    # Let's try a compact arrangement:
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons  
    # Row 2: 3 hexagons
    # Row 3: 2 hexagons
    # That's 12 total
    
    # Actually, let's use a known optimal or near-optimal packing
    # For 12 unit hexagons, a good arrangement is two rows of 6, offset
    
    # Let's try: arrange in a rectangular-like grid with hexagonal offset
    # 4 columns x 3 rows with hexagonal offset
    
    inner_hexagons = []
    
    # Hexagonal grid spacing: for unit hexagons, horizontal spacing is 3, vertical is sqrt(3)
    # Wait, for regular hexagons with side 1, the distance between centers in a hex grid:
    # Adjacent centers are distance 2*apothem = sqrt(3) apart? No.
    # For unit hexagons (side=1), the distance from center to vertex is 1.
    # In a hex grid, adjacent hexagons have centers separated by 2*side*sqrt(3)/2 = sqrt(3)
    
    # Let me reconsider. For regular hexagons with side length s=1:
    # The distance from center to any vertex is s = 1
    # The distance from center to any edge (apothem) is s*sqrt(3)/2 = sqrt(3)/2
    # In a hexagonal packing, adjacent hexagon centers are separated by 2*apothem = sqrt(3)
    
    # Let's place hexagons in a hexagonal grid pattern
    # Using axial coordinates
    
    # Positions for 12 hexagons in a compact cluster:
    # Using cube coordinates or axial (q, r)
    # Let's use a 3x4 arrangement with offset
    
    positions = []
    # Row 0 (y=0): x = 0, 1, 2, 3
    for q in range(4):
        positions.append((q, 0))
    # Row 1 (y=1): x = 0.5, 1.5, 2.5, 3.5 -> offset by 0.5
    for q in range(4):
        positions.append((q + 0.5, 1))
    # Row 2 (y=2): x = 0, 1, 2, 3
    for q in range(3):
        positions.append((q, 2))
    
    # That's 4+4+3 = 11, need one more
    # Add one more at row 2
    positions.append((3, 2))
    
    # Convert to Cartesian coordinates
    # For hex grid with pointy-top orientation:
    # x = q * sqrt(3) + r * sqrt(3)/2
    # y = r * 1.5  (for pointy top)
    
    # Actually, let's use flat-top hexagons for simplicity
    # For flat-top hexagons with side 1:
    # Width = 2, Height = sqrt(3)
    # Horizontal spacing: 1.5 * side = 1.5
    # Vertical spacing: sqrt(3) * side = sqrt(3)
    # Offset for odd columns: sqrt(3)/2
    
    # Let me just use a simple grid and compute the bounding hexagon
    
    # Simpler approach: place hexagons in a straight line first, then optimize
    # Or use a known good configuration
    
    # Let's try: 12 hexagons in a 2x6 rectangular arrangement
    # With hexagonal offset
    
    hex_centers = []
    for row in range(2):
        for col in range(6):
            x = col * 1.5
            y = row * np.sqrt(3)
            if row % 2 == 1:
                x += 0.75
            hex_centers.append((x, y))
    
    # All hexagons have angle 0 (flat-top)
    inner = []
    for (x, y) in hex_centers:
        inner.append([x, y, 0.0])
    
    # Compute bounding hexagon
    # Find the minimum side length regular hexagon that contains all inner hexagons
    # This is a non-trivial optimization problem
    
    # Simple approach: center the bounding hexagon at the centroid of inner hexagons
    # and find the minimum side length
    
    xs = [p[0] for p in inner]
    ys = [p[1] for p in inner]
    cx = np.mean(xs)
    cy = np.mean(ys)
    
    # For a regular hexagon centered at (cx, cy) with side s and angle theta,
    # we need all inner hexagon vertices to be inside
    
    # Let's do a simple search over side length and angle
    best_side = None
    best_angle = 0.0
    
    # Try different angles
    for angle_deg in range(0, 60, 5):
        # Binary search on side length
        lo = 0.0
        hi = 20.0
        
        for _ in range(30):
            mid = (lo + hi) / 2
            outer = (cx, cy, mid, float(angle_deg))
            
            valid = True
            for (ix, iy, iangle) in inner:
                if not contains_hexagon_in_outer((ix, iy, 1.0, iangle), outer):
                    valid = False
                    break
            
            if valid:
                hi = mid
                if best_side is None or mid < best_side:
                    best_side = mid
                    best_angle = float(angle_deg)
            else:
                lo = mid
        
        # Also check if this angle gives a better result
        if best_side is not None and hi < best_side:
            best_side = hi
            best_angle = float(angle_deg)
    
    if best_side is None:
        best_side = 10.0
        best_angle = 0.0
    
    return inner, (cx, cy), float(best_side), float(best_angle)

# EVOLVE-BLOCK-START
def construct():
    """
    Place 12 disjoint unit hexagons inside a regular hexagon, minimizing outer side length.
    Returns: (inner, center, side, angle)
    """
    # Use a compact hexagonal arrangement
    # 12 hexagons: center + 6 around + 5 more
    
    # Let's use a 3-4-3-2 pattern or similar
    # Actually, let's use two rows of 6 with hexagonal offset
    
    hex_centers = []
    # Row 0: 6 hexagons
    for col in range(6):
        x = col * 1.5
        y = 0.0
        hex_centers.append((x, y))
    # Row 1: 6 hexagons, offset
    for col in range(6):
        x = col * 1.5 + 0.75
        y = np.sqrt(3)
        hex_centers.append((x, y))
    
    inner = []
    for (x, y) in hex_centers:
        inner.append([x, y, 0.0])
    
    # Center the arrangement
    xs = [p[0] for p in inner]
    ys = [p[1] for p in inner]
    cx = np.mean(xs)
    cy = np.mean(ys)
    
    # Translate all to center at origin
    inner_centered = []
    for (x, y, a) in inner:
        inner_centered.append([x - cx, y - cy, a])
    
    # Find minimum bounding hexagon
    best_side = None
    best_angle = 0.0
    
    for angle_deg in range(0, 60, 3):
        lo = 0.0
        hi = 20.0
        
        for _ in range(25):
            mid = (lo + hi) / 2
            outer = (0.0, 0.0, mid, float(angle_deg))
            
            valid = True
            for (ix, iy, iangle) in inner_centered:
                if not contains_hexagon_in_outer((ix, iy, 1.0, iangle), outer):
                    valid = False
                    break
            
            if valid:
                hi = mid
                if best_side is None or mid < best_side:
                    best_side = mid
                    best_angle = float(angle_deg)
            else:
                lo = mid
        
        if best_side is not None and hi < best_side:
            best_side = hi
            best_angle = float(angle_deg)
    
    if best_side is None:
        best_side = 10.0
        best_angle = 0.0
    
    return inner_centered, (0.0, 0.0), float(best_side), float(best_angle)
# EVOLVE-BLOCK-END
