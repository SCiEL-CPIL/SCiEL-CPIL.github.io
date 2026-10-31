import math
import time

N = 11
DEADLINE = 115.0  # safety margin before 120s hard limit
START_TIME = time.time()

def deadline_reached():
    return (time.time() - START_TIME) > DEADLINE

def hex_vertices(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon with given center, side, rotation."""
    a = math.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = a + k * math.pi / 3.0
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return verts

def point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon with center (cx,cy), side, rotation."""
    # Transform to local coordinates
    dx = px - cx
    dy = py - cy
    a = math.radians(angle_deg)
    ca, sa = math.cos(-a), math.sin(-a)
    lx = dx * ca - dy * sa
    ly = dx * sa + dy * ca
    # Regular hexagon with side s centered at origin, flat-top orientation:
    # |lx| <= s, |ly| <= s * sqrt(3)/2, |lx| + |ly|/sqrt(3) <= s
    s = side
    return abs(lx) <= s + 1e-9 and abs(ly) <= s * math.sqrt(3) / 2 + 1e-9 and \
           abs(lx) + abs(ly) / math.sqrt(3) <= s + 1e-9

def hex_dist(cx1, cy1, a1, cx2, cy2, a2):
    """Minimum distance between two unit hexagons (side=1). 0 if overlapping."""
    # Approximate: check all vertex-vertex and edge-edge distances
    v1 = hex_vertices(cx1, cy1, 1.0, a1)
    v2 = hex_vertices(cx2, cy2, 1.0, a2)
    min_d = float('inf')
    for i in range(6):
        for j in range(6):
            dx = v1[i][0] - v2[j][0]
            dy = v1[i][1] - v2[j][1]
            d = math.hypot(dx, dy)
            if d < min_d:
                min_d = d
    # Also check vertex-to-edge distances for tighter bound
    for i in range(6):
        v1a = v1[i]
        v1b = v1[(i + 1) % 6]
        for j in range(6):
            v2p = v2[j]
            # Distance from point v2p to segment v1a-v1b
            ax, ay = v1a
            bx, by = v1b
            px, py = v2p
            dx, dy = bx - ax, by - ay
            if dx == 0 and dy == 0:
                d = math.hypot(px - ax, py - ay)
            else:
                t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
                t = max(0.0, min(1.0, t))
                qx = ax + t * dx
                qy = ay + t * dy
                d = math.hypot(px - qx, py - qy)
            if d < min_d:
                min_d = d
    for i in range(6):
        v2a = v2[i]
        v2b = v2[(i + 1) % 6]
        for j in range(6):
            v1p = v1[j]
            ax, ay = v2a
            bx, by = v2b
            px, py = v1p
            dx, dy = bx - ax, by - ay
            if dx == 0 and dy == 0:
                d = math.hypot(px - ax, py - ay)
            else:
                t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
                t = max(0.0, min(1.0, t))
                qx = ax + t * dx
                qy = ay + t * dy
                d = math.hypot(px - qx, py - qy)
            if d < min_d:
                min_d = d
    return min_d

def fits_in_outer(inner, outer_center, outer_side, outer_angle):
    """Check if all inner hexagons fit inside the outer hexagon."""
    ocx, ocy = outer_center
    # Outer hexagon vertices
    ov = hex_vertices(ocx, ocy, outer_side, outer_angle)
    for (ix, iy, ia) in inner:
        # Check all 6 vertices of inner hex are inside outer
        iv = hex_vertices(ix, iy, 1.0, ia)
        for (vx, vy) in iv:
            if not point_in_hex(vx, vy, ocx, ocy, outer_side, outer_angle):
                return False
    return True

def no_overlap(inner):
    """Check that all pairs of inner hexagons are disjoint."""
    n = len(inner)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1, a1 = inner[i]
            x2, y2, a2 = inner[j]
            if hex_dist(x1, y1, a1, x2, y2, a2) < -1e-9:
                return False
    return True

def valid_config(inner, outer_center, outer_side, outer_angle):
    return fits_in_outer(inner, outer_center, outer_side, outer_angle) and no_overlap(inner)

def construct():
    """Construct N unit hexagons inside a regular hexagon, minimizing outer side."""
    # Hexagonal lattice arrangement: center + rings
    # For N=11: 1 + 6 + 4 (partial ring 3) or use 1+6+4
    # Standard hex packing: positions in a hex grid
    # Ring 0: (0,0)
    # Ring 1: 6 positions at distance 2 (for unit hexes, spacing = 2 for same orientation)
    # Actually for unit hexes with side 1, flat-top, the center-to-center distance for non-overlapping
    # same-orientation hexes is 2 (touching flat sides) or sqrt(3) (touching corners)
    
    # Use a hexagonal lattice with spacing 2 (same orientation, flat sides touching)
    # This gives a compact arrangement.
    
    # Hex grid coordinates: axial coords
    # For N=11, use: center(1) + ring1(6) + ring2 partial(4)
    # Ring 2 has 12 positions, pick 4
    
    # Same-orientation hexes on hex grid with spacing 2:
    # Axial: (q, r) -> (2*sqrt(3)/2 * (q + r/2), 2 * 3/2 * r) = (sqrt(3)*(q+r/2), 3*r/2) * 2... 
    # Let me use: for same-orientation unit hexes, center-to-center for adjacent (flat side to flat side) = 2
    # Axial coords (q,r): x = 2*(q + r/2) * (sqrt(3)/2) * 2 = ... 
    # Simpler: for flat-top hex, neighbors are at distance 2 in 6 directions:
    # (2,0), (1,sqrt(3)), (-1,sqrt(3)), (-2,0), (-1,-sqrt(3)), (1,-sqrt(3))
    
    sqrt3 = math.sqrt(3)
    
    # Generate hex grid positions for rings 0, 1, 2
    positions = [(0.0, 0.0)]
    # Ring 1: 6 positions
    for k in range(6):
        angle = math.radians(60 * k)
        positions.append((2.0 * math.cos(angle), 2.0 * math.sin(angle)))
    # Ring 2: 12 positions, pick 4 to get total 11
    ring2_positions = []
    for k in range(6):
        angle = math.radians(60 * k)
        ring2_positions.append((4.0 * math.cos(angle), 4.0 * math.sin(angle)))
    for k in range(6):
        angle = math.radians(60 * k + 30)
        ring2_positions.append((2.0 * sqrt3 * math.cos(angle), 2.0 * sqrt3 * math.sin(angle)))
    
    # Pick 4 from ring 2 (the ones closest to existing to form compact shape)
    # Just take first 4
    positions.extend(ring2_positions[:4])
    
    # Ensure we have exactly N
    positions = positions[:N]
    
    inner = [[p[0], p[1], 0.0] for p in positions]
    
    # Now find minimal outer hexagon that contains all
    # Try several outer hexagon angles and centers
    best_side = float('inf')
    best_outer = ((0.0, 0.0), 40.0, 0.0)
    
    # The outer hexagon should be centered near the centroid of the inner hexagons
    cx = sum(p[0] for p in positions) / len(positions)
    cy = sum(p[1] for p in positions) / len(positions)
    
    # Try different outer hexagon rotations
    for outer_angle in [0, 30, 60, 90, 120, 150]:
        # Compute required side for this orientation
        # For a regular hexagon with center (cx,cy) and angle, find minimum side
        # that contains all inner hex vertices
        
        # Binary search on side length
        lo, hi = 0.0, 50.0
        for _ in range(50):
            mid = (lo + hi) / 2
            if fits_in_outer(inner, (cx, cy), mid, float(outer_angle)):
                hi = mid
            else:
                lo = mid
        if hi < best_side:
            best_side = hi
            best_outer = ((cx, cy), hi, float(outer_angle))
    
    ocx, ocy, oside, oangle = best_outer
    return inner, (ocx, ocy), oside, oangle
