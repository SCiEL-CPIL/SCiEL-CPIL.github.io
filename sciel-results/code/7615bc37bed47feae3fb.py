"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time

N = 12

# EVOLVE-BLOCK-START
def _hexagon_vertices(center_x, center_y, side, angle_deg):
    """Return vertices of a regular hexagon with given center, side length, and rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        x = center_x + side * math.cos(theta)
        y = center_y + side * math.sin(theta)
        verts.append((x, y))
    return verts

def _point_in_hexagon(px, py, center_x, center_y, side, angle_deg):
    """Check if point (px,py) is inside hexagon with given params."""
    # Transform point to hexagon's local coordinate system
    dx = px - center_x
    dy = py - center_y
    angle_rad = math.radians(angle_deg)
    # Rotate by -angle to align hexagon with axes
    cos_a = math.cos(-angle_rad)
    sin_a = math.sin(-angle_rad)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    # Regular hexagon with flat top (angle 0): vertices at angles 0, 60, 120, 180, 240, 300
    # In local coords, the hexagon has vertices at (side*cos(k*60), side*sin(k*60))
    # A point is inside if it satisfies all half-plane constraints
    # The hexagon can be described as intersection of 6 half-planes
    # For a hexagon centered at origin with side length s, the distance from center to each side is s*sqrt(3)/2
    # The normal directions are at angles 30, 90, 150, 210, 270, 330 degrees
    
    apothem = side * math.sqrt(3) / 2.0
    for i in range(6):
        normal_angle = math.radians(30 + i * 60)
        nx = math.cos(normal_angle)
        ny = math.sin(normal_angle)
        # The half-plane constraint: nx*lx + ny*ly <= apothem
        if nx * lx + ny * ly > apothem + 1e-9:
            return False
    return True

def _hexagons_disjoint(hex1, hex2):
    """Check if two hexagons are disjoint (no overlap). hex = (cx, cy, angle_deg, side)"""
    cx1, cy1, a1, s1 = hex1
    cx2, cy2, a2, s2 = hex2
    
    # First, quick bounding box check
    # Max extent of a hexagon is its side length * 2 (diameter)
    max_extent = max(s1, s2) * 2.0
    dx = abs(cx1 - cx2)
    dy = abs(cy1 - cy2)
    if dx > max_extent * 2 or dy > max_extent * 2:
        return True
    
    # Check if any vertex of hex1 is inside hex2 or vice versa
    verts1 = _hexagon_vertices(cx1, cy1, s1, a1)
    verts2 = _hexagon_vertices(cx2, cy2, s2, a2)
    
    for vx, vy in verts1:
        if _point_in_hexagon(vx, vy, cx2, cy2, s2, a2):
            return False
    
    for vx, vy in verts2:
        if _point_in_hexagon(vx, vy, cx1, cy1, s1, a1):
            return False
    
    # Check edge intersections
    for i in range(6):
        x1, y1 = verts1[i]
        x2, y2 = verts1[(i + 1) % 6]
        for j in range(6):
            x3, y3 = verts2[j]
            x4, y4 = verts2[(j + 1) % 6]
            if _segments_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
                return False
    
    return True

def _segments_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    """Check if two line segments intersect."""
    def _cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    p1 = (x1, y1)
    p2 = (x2, y2)
    p3 = (x3, y3)
    p4 = (x4, y4)
    
    d1 = _cross(p3, p4, p1)
    d2 = _cross(p3, p4, p2)
    d3 = _cross(p1, p2, p3)
    d4 = _cross(p1, p2, p4)
    
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    
    # Check collinear cases
    if d1 == 0 and _on_segment(p3, p4, p1):
        return True
    if d2 == 0 and _on_segment(p3, p4, p2):
        return True
    if d3 == 0 and _on_segment(p1, p2, p3):
        return True
    if d4 == 0 and _on_segment(p1, p2, p4):
        return True
    
    return False

def _on_segment(p, q, r):
    """Check if point q lies on segment pr."""
    if min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and \
       min(p[1], r[1]) <= q[1] <= max(p[1], r[1]):
        return True
    return False

def _min_outer_side(inner_positions, outer_center, outer_angle_deg):
    """Find minimum outer hexagon side length that contains all inner hexagons."""
    cx, cy = outer_center
    oa = outer_angle_deg
    
    # For each inner hexagon, find the required outer side length
    # The outer hexagon must contain all vertices of all inner hexagons
    # For a regular hexagon with center (cx,cy), side s, angle a, a point (px,py) is inside
    # if for all 6 sides: normal_i . (p - center) <= apothem
    
    max_required = 0.0
    for (ix, iy, iangle) in inner_positions:
        verts = _hexagon_vertices(ix, iy, 1.0, iangle)
        for (vx, vy) in verts:
            # Transform to outer hexagon's local frame
            dx = vx - cx
            dy = vy - cy
            angle_rad = math.radians(oa)
            cos_a = math.cos(-angle_rad)
            sin_a = math.sin(-angle_rad)
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            
            # Find max projection onto normals
            max_proj = 0.0
            for i in range(6):
                normal_angle = math.radians(30 + i * 60)
                nx = math.cos(normal_angle)
                ny = math.sin(normal_angle)
                proj = nx * lx + ny * ly
                if proj > max_proj:
                    max_proj = proj
            
            # Required apothem is max_proj, so required side is max_proj / (sqrt(3)/2)
            required_side = max_proj / (math.sqrt(3) / 2.0)
            if required_side > max_required:
                max_required = required_side
    
    return max_required

def _is_valid_configuration(inner_positions, outer_center, outer_side, outer_angle_deg):
    """Check if configuration is valid: all inner hexagons are inside outer, and disjoint from each other."""
    cx, cy = outer_center
    
    # Check all inner hexagons are inside outer
    for (ix, iy, iangle) in inner_positions:
        verts = _hexagon_vertices(ix, iy, 1.0, iangle)
        for (vx, vy) in verts:
            if not _point_in_hexagon(vx, vy, cx, cy, outer_side, outer_angle_deg):
                return False
    
    # Check all pairs are disjoint
    n = len(inner_positions)
    for i in range(n):
        for j in range(i + 1, n):
            hex1 = (inner_positions[i][0], inner_positions[i][1], inner_positions[i][2], 1.0)
            hex2 = (inner_positions[j][0], inner_positions[j][1], inner_positions[j][2], 1.0)
            if not _hexagons_disjoint(hex1, hex2):
                return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_result = None
    best_side = float('inf')
    
    # Strategy: Try different arrangements of 12 hexagons
    # A natural arrangement is a honeycomb pattern
    
    # Generate candidate configurations
    candidates = []
    
    # Candidate 1: Honeycomb arrangement in a 3x4 grid pattern
    # Hexagons in a honeycomb: row offset
    # For side length 1, horizontal spacing = 3, vertical spacing = 3*sqrt(3)/2
    # Actually for flat-top hexagons: horizontal dist = 3, vertical = 3*sqrt(3)/2
    # Let me think more carefully.
    
    # A regular hexagon with side 1 has:
    # Width (flat to flat) = sqrt(3)
    # Height (point to point) = 2
    # For flat-top orientation (angle 0):
    #   The hexagon has vertices at angles 0, 60, 120, 180, 240, 300
    #   So it's "pointy" left and right, flat top and bottom? No...
    #   At angle 0: vertex at (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5,-sqrt(3)/2), (0.5,-sqrt(3)/2)
    #   So it's pointy left-right, flat top-bottom.
    
    # For tiling with flat-top hexagons:
    # Horizontal spacing between centers: 3 (since width is 2, plus 1 gap... no)
    # Actually for adjacent flat-top hexagons sharing a vertical edge:
    # The shared edge is vertical. Distance between centers = 3? 
    # Let me just use known honeycomb spacing.
    
    # For unit hexagons in a honeycomb:
    # dx = 3, dy = 3*sqrt(3)/2 for flat-top
    # Or dx = 3*sqrt(3)/2, dy = 3 for pointy-top
    
    sqrt3 = math.sqrt(3)
    
    # Try honeycomb with flat-top hexagons (angle 0)
    # Centers at:
    # Row r, col c: x = c * 3 + (r % 2) * 1.5, y = r * 3 * sqrt3 / 2
    # But we need 12 hexagons. Let's try a 4x3 or 3x4 arrangement.
    
    # Try 4 columns x 3 rows
    for rot in [0, 30, 60, 90, 120, 150]:
        positions = []
        # 3 rows, 4 cols
        for r in range(3):
            for c in range(4):
                x = c * 3.0 + (r % 2) * 1.5
                y = r * 3 * sqrt3 / 2
                positions.append([x, y, float(rot)])
        
        # Center the arrangement
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        for p in positions:
            p[0] -= cx
            p[1] -= cy
        
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        side = _min_outer_side(positions, outer_center, outer_angle)
        
        if _is_valid_configuration(positions, outer_center, side, outer_angle):
            if side < best_side:
                best_side = side
                best_result = (positions, outer_center, side, outer_angle)
        
        if time.time() > deadline:
            break
    
    # Try other arrangements: 2x6
    for rot in [0, 30, 60, 90, 120, 150]:
        positions = []
        for r in range(2):
            for c in range(6):
                x = c * 3.0 + (r % 2) * 1.5
                y = r * 3 * sqrt3 / 2
                positions.append([x, y, float(rot)])
        
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        for p in positions:
            p[0] -= cx
            p[1] -= cy
        
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        side = _min_outer_side(positions, outer_center, outer_angle)
        
        if _is_valid_configuration(positions, outer_center, side, outer_angle):
            if side < best_side:
                best_side = side
                best_result = (positions, outer_center, side, outer_angle)
        
        if time.time() > deadline:
            break
    
    # Try circular arrangement
    for rot in [0, 30, 60, 90, 120, 150]:
        positions = []
        # 12 hexagons in a circle
        radius = 3.0  # Distance from center to inner hexagon centers
        for i in range(12):
            angle = i * math.pi / 6.0
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            positions.append([x, y, float(rot)])
        
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        side = _min_outer_side(positions, outer_center, outer_angle)
        
        if _is_valid_configuration(positions, outer_center, side, outer_angle):
            if side < best_side:
                best_side = side
                best_result = (positions, outer_center, side, outer_angle)
        
        if time.time() > deadline:
            break
    
    # Try a more compact honeycomb: hexagonal cluster of 12
    # A hexagonal number pattern: 1 + 6 + 5 = 12? No, hexagonal numbers are 1, 7, 19...
    # Let's just try a 2-3-4-3 arrangement
    for rot in [0, 30, 60, 90, 120, 150]:
        positions = []
        # Row 0: 2 hexagons
        # Row 1: 3 hexagons
        # Row 2: 4 hexagons
        # Row 3: 3 hexagons
        row_counts = [2, 3, 4, 3]
        for r, count in enumerate(row_counts):
            offset = (4 - count) / 2.0 * 3.0  # Center each row
            for c in range(count):
                x = offset + c * 3.0 + (r % 2) * 1.5
                y = r * 3 * sqrt3 / 2
                positions.append([x, y, float(rot)])
        
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        for p in positions:
            p[0] -= cx
            p[1] -= cy
        
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        side = _min_outer_side(positions, outer_center, outer_angle)
        
        if _is_valid_configuration(positions, outer_center, side, outer_angle):
            if side < best_side:
                best_side = side
                best_result = (positions, outer_center, side, outer_angle)
        
        if time.time() > deadline:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: return a valid configuration with a large outer hexagon
    positions = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return positions, (0.0, 0.0), 40.0, 0.0
# EVOLVE-BLOCK-END
