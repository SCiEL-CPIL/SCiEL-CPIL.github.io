"""Optimized hexagon packing: 11 unit hexagons in compact hexagonal lattice."""
import math
import time

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    verts = []
    a = math.radians(angle_deg)
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return verts

def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    # Transform to local coords
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    dx, dy = px - cx, py - cy
    lx = dx * ca + dy * sa
    ly = -dx * sa + dy * ca
    # For a regular hexagon with side s, the distance from center to each edge is s*sqrt(3)/2
    # The hexagon is the intersection of 6 half-planes
    h = side * math.sqrt(3) / 2.0
    # Check 6 edges
    for i in range(6):
        theta = i * math.pi / 3.0
        # Normal direction for edge i
        nx = math.cos(theta)
        ny = math.sin(theta)
        dist = lx * nx + ly * ny
        if dist > h + 1e-9:
            return False
    return True

def _segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4."""
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
    """Check if two hexagons (lists of vertices) are disjoint (no overlap, no touching)."""
    v1, v2 = hex1, hex2
    # Check if any vertex of one is inside the other
    for v in v1:
        if _point_in_hex(v[0], v[1], *[_center_from_hex(v2)]):
            return False
    for v in v2:
        if _point_in_hex(v[0], v[1], *[_center_from_hex(v1)]):
            return False
    # Check edge intersections
    for i in range(6):
        for j in range(6):
            if _segments_intersect(v1[i], v1[(i+1)%6], v2[j], v2[(j+1)%6]):
                return False
    return True

def _center_from_hex(verts):
    """Get center and side from hex vertices."""
    cx = sum(v[0] for v in verts) / 6.0
    cy = sum(v[1] for v in verts) / 6.0
    # Side length is distance from center to vertex
    side = math.sqrt((verts[0][0]-cx)**2 + (verts[0][1]-cy)**2)
    # Angle
    angle = math.degrees(math.atan2(verts[0][1]-cy, verts[0][0]-cx)) % 60.0
    return cx, cy, side, angle

def _all_disjoint(inner_hexes, eps=1e-9):
    """Check all pairs of inner hexagons are disjoint."""
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i+1, n):
            if not _hexagons_disjoint(inner_hexes[i], inner_hexes[j], eps):
                return False
    return True

def _inner_in_outer(inner_hexes, outer_center, outer_side, outer_angle, eps=1e-9):
    """Check all inner hexagons are inside the outer hexagon."""
    for hex_verts in inner_hexes:
        for v in hex_verts:
            if not _point_in_hex(v[0], v[1], outer_center[0], outer_center[1], outer_side, outer_angle, eps):
                return False
    return True

def _min_outer_side(inner_hexes, outer_angle, center_search=True):
    """Find minimum outer side length for given inner hexagons and outer angle."""
    # First, find bounding box of all vertices
    all_verts = []
    for hex_verts in inner_hexes:
        all_verts.extend(hex_verts)
    
    if not all_verts:
        return 0.0, (0.0, 0.0)
    
    # Compute centroid as initial center guess
    cx0 = sum(v[0] for v in all_verts) / len(all_verts)
    cy0 = sum(v[1] for v in all_verts) / len(all_verts)
    
    best_side = float('inf')
    best_center = (cx0, cy0)
    
    # Try different centers around the centroid
    if center_search:
        # Search over a grid of centers
        max_extent = max(max(v[0] for v in all_verts) - min(v[0] for v in all_verts),
                        max(v[1] for v in all_verts) - min(v[1] for v in all_verts))
        search_range = max_extent * 0.5
        n_grid = 20
        for i in range(n_grid):
            for j in range(n_grid):
                cx = cx0 + search_range * (2*i/(n_grid-1) - 1) if n_grid > 1 else cx0
                cy = cy0 + search_range * (2*j/(n_grid-1) - 1) if n_grid > 1 else cy0
                # For this center, find min side length
                # The min side is the max over all vertices of the "hex distance"
                a = math.radians(outer_angle)
                ca, sa = math.cos(a), math.sin(a)
                max_dist = 0.0
                for v in all_verts:
                    dx, dy = v[0] - cx, v[1] - cy
                    lx = dx * ca + dy * sa
                    ly = -dx * sa + dy * ca
                    # For a regular hexagon with side s, the constraint is:
                    # |lx * cos(k*pi/3) + ly * sin(k*pi/3)| <= s*sqrt(3)/2 for all k
                    # So s >= max_k |lx*cos(k*pi/3) + ly*sin(k*pi/3)| * 2/sqrt(3)
                    for k in range(6):
                        theta = k * math.pi / 3.0
                        val = abs(lx * math.cos(theta) + ly * math.sin(theta))
                        dist_needed = val * 2.0 / math.sqrt(3.0)
                        if dist_needed > max_dist:
                            max_dist = dist_needed
                if max_dist < best_side:
                    best_side = max_dist
                    best_center = (cx, cy)
    else:
        cx, cy = cx0, cy0
        a = math.radians(outer_angle)
        ca, sa = math.cos(a), math.sin(a)
        max_dist = 0.0
        for v in all_verts:
            dx, dy = v[0] - cx, v[1] - cy
            lx = dx * ca + dy * sa
            ly = -dx * sa + dy * ca
            for k in range(6):
                theta = k * math.pi / 3.0
                val = abs(lx * math.cos(theta) + ly * math.sin(theta))
                dist_needed = val * 2.0 / math.sqrt(3.0)
                if dist_needed > max_dist:
                    max_dist = dist_needed
        best_side = max_dist
        best_center = (cx, cy)
    
    return best_side, best_center

def construct():
    """Construct 11 unit hexagons in optimal hexagonal lattice arrangement."""
    start_time = time.time()
    deadline = 120.0  # Safety margin
    
    # Hexagonal lattice positions for 11 hexagons
    # Center at origin, using axial coordinates
    # Ring 0: (0,0) - 1 hex
    # Ring 1: 6 hexes at distance sqrt(3)
    # Ring 2: need 4 more from the 12 positions in ring 2
    
    # In a hexagonal lattice with spacing d = sqrt(3) (distance between centers of adjacent unit hexagons)
    # Axial coordinate (q, r) maps to:
    # x = d * (q + r/2)
    # y = d * (r * sqrt(3)/2)
    
    d = math.sqrt(3)  # Center-to-center distance for adjacent unit hexagons
    
    # Positions in axial coordinates (q, r)
    # For 11 hexagons: center + 6 in ring 1 + 4 in ring 2
    # Ring 1: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    # Ring 2: choose 4 that give most compact shape
    
    axial_positions = [
        (0, 0),       # Center
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),  # Ring 1
        (2, 0), (2, -1), (1, -2), (-1, 1),  # 4 from ring 2 - choose compact subset
    ]
    
    # Convert to Cartesian
    positions = []
    for q, r in axial_positions:
        x = d * (q + r / 2.0)
        y = d * (r * math.sqrt(3) / 2.0)
        positions.append((x, y))
    
    # All hexagons have the same orientation (0 degrees) for optimal packing
    angle_deg = 0.0
    inner = [[pos[0], pos[1], angle_deg] for pos in positions]
    
    # Generate hexagon vertices for validation
    inner_hexes = [_hex_vertices(p[0], p[1], 1.0, p[2]) for p in inner]
    
    # Validate: all inner hexagons must be disjoint
    if not _all_disjoint(inner_hexes):
        # Fallback: spread them out more
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Find optimal outer hexagon
    best_overall_side = float('inf')
    best_overall_center = (0.0, 0.0)
    best_overall_angle = 0.0
    
    # Try different outer angles (0, 30 degrees are the two symmetric cases)
    for outer_angle in [0.0, 30.0]:
        side, center = _min_outer_side(inner_hexes, outer_angle, center_search=True)
        if side < best_overall_side:
            best_overall_side = side
            best_overall_center = center
            best_overall_angle = outer_angle
        
        if time.time() - start_time > deadline - 5:
            break
    
    # Verify the solution is valid
    outer_center = best_overall_center
    outer_side = best_overall_side
    outer_angle = best_overall_angle
    
    if not _inner_in_outer(inner_hexes, outer_center, outer_side, outer_angle):
        # Increase side slightly to ensure containment
        outer_side *= 1.001
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
