"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

N = 11

# EVOLVE-BLOCK-START
def _hex_corners(cx, cy, side, angle_deg):
    """Return list of 6 corners of a regular hexagon."""
    angle_rad = math.radians(angle_deg)
    corners = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        corners.append((x, y))
    return corners

def _point_in_polygon(point, polygon, epsilon=1e-9):
    """Ray casting algorithm for point in polygon."""
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi - epsilon):
            inside = not inside
        j = i
    return inside

def _segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4 (proper intersection)."""
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

def _hexagons_disjoint(hex1, hex2, epsilon=1e-9):
    """Check if two hexagons (lists of corners) are disjoint (no overlap)."""
    c1 = hex1
    c2 = hex2
    
    # Check if any vertex of one is inside the other
    for v in c1:
        if _point_in_polygon(v, c2, epsilon):
            return False
    for v in c2:
        if _point_in_polygon(v, c1, epsilon):
            return False
    
    # Check if any edges intersect
    for i in range(6):
        j = (i + 1) % 6
        for k in range(6):
            l = (k + 1) % 6
            if _segments_intersect(c1[i], c1[j], c2[k], c2[l]):
                return False
    
    return True

def _min_enclosing_hex_side(points, angles_to_try):
    """Find the minimum side length of a regular hexagon (among given rotations)
    that contains all points."""
    min_side = float('inf')
    for angle in angles_to_try:
        # The 6 normal directions for a hexagon with rotation angle
        # Vertices at angle, angle+60, ...
        # Normals at angle+30, angle+90, ...
        normals = []
        for i in range(6):
            angle_rad = math.radians(angle + 30 + i * 60)
            normals.append((math.cos(angle_rad), math.sin(angle_rad)))
        
        # For each normal, get min/max projection
        projections = []
        for nx, ny in normals:
            vals = [x * nx + y * ny for x, y in points]
            projections.append((min(vals), max(vals)))
        
        # Determine center (cx, cy)
        # Using first two normals
        nx0, ny0 = normals[0]
        nx1, ny1 = normals[1]
        b0 = (projections[0][0] + projections[0][1]) / 2
        b1 = (projections[1][0] + projections[1][1]) / 2
        
        det = nx0 * ny1 - nx1 * ny0
        if abs(det) < 1e-12:
            continue
        
        cx = (b0 * ny1 - b1 * ny0) / det
        cy = (nx0 * b1 - nx1 * b0) / det
        
        # Calculate required apothem
        apothem = 0
        for i in range(6):
            nx, ny = normals[i]
            center_proj = cx * nx + cy * ny
            min_proj = projections[i][0]
            max_proj = projections[i][1]
            dist_min = abs(min_proj - center_proj)
            dist_max = abs(max_proj - center_proj)
            apothem = max(apothem, dist_min, dist_max)
        
        side = apothem * 2 / math.sqrt(3)
        if side < min_side:
            min_side = side
            # Store best center and angle for reconstruction if needed, 
            # but we just need the side length for score, and we can return 
            # the center/angle that achieved it.
            # However, the interface requires returning the specific center and angle.
            # So we should return (side, cx, cy, angle) or similar.
            # Let's modify to return the full tuple.
            pass
            
    # We need to return the best (side, cx, cy, angle)
    # Let's refactor to track the best.
    best = (float('inf'), 0, 0, 0)
    for angle in angles_to_try:
        normals = []
        for i in range(6):
            angle_rad = math.radians(angle + 30 + i * 60)
            normals.append((math.cos(angle_rad), math.sin(angle_rad)))
        
        projections = []
        for nx, ny in normals:
            vals = [x * nx + y * ny for x, y in points]
            projections.append((min(vals), max(vals)))
        
        nx0, ny0 = normals[0]
        nx1, ny1 = normals[1]
        b0 = (projections[0][0] + projections[0][1]) / 2
        b1 = (projections[1][0] + projections[1][1]) / 2
        
        det = nx0 * ny1 - nx1 * ny0
        if abs(det) < 1e-12:
            continue
        
        cx = (b0 * ny1 - b1 * ny0) / det
        cy = (nx0 * b1 - nx1 * b0) / det
        
        apothem = 0
        for i in range(6):
            nx, ny = normals[i]
            center_proj = cx * nx + cy * ny
            min_proj = projections[i][0]
            max_proj = projections[i][1]
            dist_min = abs(min_proj - center_proj)
            dist_max = abs(max_proj - center_proj)
            apothem = max(apothem, dist_min, dist_max)
        
        side = apothem * 2 / math.sqrt(3)
        if side < best[0]:
            best = (side, cx, cy, angle)
            
    return best

def construct():
    """Place N unit hexagons inside a regular hexagon, minimizing outer side length."""
    deadline = time.time() + 120.0
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = None
    best_outer_angle = None
    
    # Candidate 1: Linear
    linear_positions = []
    for k in range(-(N // 2), N - N // 2):
        linear_positions.append((2.0 * k, 0.0))
        
    # Candidate 2: Hexagonal Cluster (1 + 6 + 4)
    # Use axial coordinates for pointy-top hexagons
    # x = sqrt(3) * (q + r/2), y = 1.5 * r
    hex_positions = []
    # Center
    hex_positions.append((0, 0))
    # Ring 1
    ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    for q, r in ring1:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        hex_positions.append((x, y))
    # Ring 2: Pick 4 that are compact
    # (2,0), (1,1), (0,2), (-1,2) -- these form a cluster on one side?
    # Better to pick symmetric ones or just any 4 that are valid.
    # Let's pick (2,0), (0,2), (-2,0), (0,-2) ? No, that's spread out.
    # Let's pick (2,0), (1,1), (-1,1), (-2,0) ?
    # Actually, just picking the first 4 from a list of ring 2 coords is fine.
    ring2_candidates = [(2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0), (-1, -1), (0, -2), (1, -2), (2, -1), (1, -1), (-1, 1)]
    # Note: (-1, 1) is in ring 1. Let's use a proper ring 2 list.
    # Ring 2 axial coords:
    # (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (1,-1) is wrong.
    # Correct ring 2:
    # (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (1,-1) is not right.
    # Let's just generate all points in a small grid and pick the first 11 that form a connected component?
    # Simpler: Just use the 7-hex flower and add 4 more at distance sqrt(3) from the ring 1.
    # For now, let's just evaluate the linear one and a "square-ish" grid one.
    
    # Candidate 3: 3x4 Grid (12 hexes, remove 1)
    grid_positions = []
    for row in range(3):
        for col in range(4):
            x = col * math.sqrt(3) + (row % 2) * math.sqrt(3) / 2
            y = row * 1.5
            grid_positions.append((x, y))
    # Remove the last one to get 11
    grid_positions_11 = grid_positions[:11]
    
    candidates = [
        linear_positions,
        hex_positions + [(math.sqrt(3)*2, 0), (math.sqrt(3)*1, 1.5), (0, 3), (-math.sqrt(3)*1, 1.5)], # Just adding 4 arbitrary ones, might overlap or be invalid
        grid_positions_11
    ]
    
    for positions in candidates:
        if len(positions) != N:
            continue
            
        # Generate all vertices of inner hexagons
        all_points = []
        inner_list = []
        for x, y in positions:
            inner_list.append((x, y, 0.0)) # angle 0
            corners = _hex_corners(x, y, 1.0, 0.0)
            all_points.extend(corners)
            
        # Find min enclosing hexagon for these points
        side, cx, cy, angle = _min_enclosing_hex_side(all_points, [0, 30, 60, 90, 120, 150])
        
        if side < best_side:
            # Validate
            # Check disjointness
            valid = True
            hexes = []
            for x, y, ang in inner_list:
                hexes.append(_hex_corners(x, y, 1.0, ang))
                
            for i in range(N):
                for j in range(i+1, N):
                    if not _hexagons_disjoint(hexes[i], hexes[j]):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Check containment
                outer_corners = _hex_corners(cx, cy, side, angle)
                for hex_corners in hexes:
                    for corner in hex_corners:
                        if not _point_in_polygon(corner, outer_corners):
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                best_side = side
                best_inner = inner_list
                best_outer_center = (cx, cy)
                best_outer_angle = angle
    
    if best_inner is None:
        # Fallback to linear with large side
        inner_list = [(2.0 * k, 0.0, 0.0) for k in range(-(N // 2), N - N // 2)]
        return inner_list, (0.0, 0.0), 12.0, 0.0
        
    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END
