"""Optimized hexagon packing: 11 unit hexagons in a compact hexagonal lattice."""
import math
import time

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


def _min_outer_side(hexes, ox, oy, oangle=0.0):
    """Find minimum side length of outer hexagon centered at (ox, oy) with rotation oangle
    that contains all hexagons in `hexes`."""
    max_val = 0.0
    a = -math.radians(oangle)
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    for cx, cy, s, ang in hexes:
        verts = _hex_vertices(cx, cy, s, ang)
        for vx, vy in verts:
            dx = vx - ox
            dy = vy - oy
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            for k in range(6):
                theta = math.radians(30 + 60 * k)
                proj = lx * math.cos(theta) + ly * math.sin(theta)
                if proj > max_val:
                    max_val = proj
    return max_val * 2.0 / math.sqrt(3)


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s limit

    # Hexagonal lattice with exact touching spacing
    # For unit hexagons (side 1), apothem = sqrt(3)/2
    # Adjacent touching hexagons have center distance = 2 * apothem = sqrt(3)
    # In axial coords (q, r) for flat-top hexagons:
    #   x = sqrt(3) * (q + r/2)
    #   y = 1.5 * r
    # This gives center-to-center distance of sqrt(3) for neighbors.

    # 11 hexagons: center + ring 1 (6) + 4 from ring 2
    # Ring 2 positions in axial coords (hex distance 2 from origin):
    # (2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-1,-2), (-2,-2), (-2,-1), (-2,0), (-2,1), (-2,2), (-1,2), (0,2), (1,2)
    # That's 14 positions. We pick 4 symmetrically.
    # For best symmetry, pick: (2,0), (0,2), (-2,2), (-2,0), (0,-2), (-2,-2) — that's 6.
    # We need exactly 4. Pick 4 that maintain symmetry: (2,0), (0,2), (-2,2), (-2,0) — 4 positions.
    # Or better: pick 4 that are spread out: (2,0), (0,2), (-2,2), (-2,0)

    # Let me use a different approach: generate all ring-2 positions and pick 4 that minimize the bounding hexagon.
    
    sqrt3 = math.sqrt(3)

    def axial_to_xy(q, r):
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        return (x, y)

    # Center
    positions = [(0, 0)]

    # Ring 1: 6 neighbors
    ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    positions.extend(ring1)

    # Ring 2: 12 positions (hex distance 2)
    # Generate by walking around
    ring2 = []
    dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    sq, sr = 0, 0
    for di, (dq, dr) in enumerate(dirs):
        for step in range(2):
            sq += dq
            sr += dr
            if (sq, sr) not in ring1 and (sq, sr) != (0, 0) and (sq, sr) not in ring2:
                ring2.append((sq, sr))

    # We need 4 more from ring2. Try all combinations of 4 from ring2 and pick the one
    # that minimizes the outer hexagon side length.
    # ring2 has 12 positions, C(12,4) = 495 combinations. Very manageable.

    from itertools import combinations

    best_side = float('inf')
    best_positions = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # Precompute xy for ring2
    ring2_xy = [axial_to_xy(q, r) for q, r in ring2]

    # Base positions (center + ring1) in xy
    base_positions = [axial_to_xy(q, r) for q, r in positions]

    # For each combination of 4 ring2 positions, compute the optimal outer hexagon
    # Since all inner hexagons are axis-aligned, the optimal outer is also axis-aligned.
    # The optimal center is the centroid of all 11 hexagons' vertices... but simpler:
    # For axis-aligned hexagons, the optimal outer hexagon center is the centroid of the
    # "extreme" points. Let's just try the centroid and a few offsets.

    for combo in combinations(range(len(ring2)), 4):
        if time.time() > deadline:
            break

        # Full set of 11 positions in xy
        full_xy = base_positions + [ring2_xy[i] for i in combo]

        # Compute centroid
        cx_mean = sum(x for x, y in full_xy) / 11
        cy_mean = sum(y for x, y in full_xy) / 11

        # Try a few candidate centers and angles
        candidates = [
            (cx_mean, cy_mean, 0.0),
            (cx_mean, cy_mean, 30.0),
            (0.0, 0.0, 0.0),
            (cx_mean + 0.1, cy_mean, 0.0),
            (cx_mean - 0.1, cy_mean, 0.0),
            (cx_mean, cy_mean + 0.1, 0.0),
            (cx_mean, cy_mean - 0.1, 0.0),
        ]

        for ocx, ocy, oang in candidates:
            # Build inner hexagons
            inner_hexes = [(x, y, 1.0, 0.0) for x, y in full_xy]
            side = _min_outer_side(inner_hexes, ocx, ocy, oang)
            if side < best_side:
                best_side = side
                best_positions = full_xy
                best_center = (ocx, ocy)
                best_angle = oang

    # Build final inner hexagons
    inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_positions]

    # Verify disjointness
    if not _all_disjoint(inner_hexes):
        # Fallback: add small gap
        sqrt3_gap = sqrt3 * 1.01
        inner_hexes = []
        for x, y in best_positions:
            inner_hexes.append((x * (sqrt3_gap / sqrt3), y * (sqrt3_gap / sqrt3), 1.0, 0.0))
        # Recompute outer
        cx_mean = sum(x for x, y, s, a in inner_hexes) / 11
        cy_mean = sum(y for x, y, s, a in inner_hexes) / 11
        best_side = _min_outer_side(inner_hexes, cx_mean, cy_mean, 0.0)
        best_center = (cx_mean, cy_mean)
        best_angle = 0.0

    # Verify all inner hexagons are inside outer
    outer_hex = (best_center[0], best_center[1], best_side, best_angle)
    valid = True
    for h in inner_hexes:
        if not _inner_in_outer(h, outer_hex):
            valid = False
            break

    if not valid:
        # Increase side slightly
        best_side *= 1.001
        outer_hex = (best_center[0], best_center[1], best_side, best_angle)

    # Format output: inner rows (x, y, angle_degrees)
    inner = [[h[0], h[1], h[3]] for h in inner_hexes]

    return inner, (best_center[0], best_center[1]), best_side, best_angle
