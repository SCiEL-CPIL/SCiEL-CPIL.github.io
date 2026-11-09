"""Pack N unit regular hexagons inside the smallest possible regular hexagon."""
import math
import time

N = 11

# EVOLVE-BLOCK-START

def _hex_points(cx, cy, r, angle):
    """Return the 6 vertex coordinates of a regular hexagon of side r centered at (cx, cy) rotated by angle radians."""
    pts = []
    for k in range(6):
        a = angle + k * math.pi / 3.0
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def _point_in_hex(px, py, cx, cy, s, angle):
    """Check if point (px, py) is inside a regular hexagon of side s centered at (cx, cy) rotated by angle."""
    # Transform to hexagon's local frame
    dx = px - cx
    dy = py - cy
    cos_a = math.cos(-angle)
    sin_a = math.sin(-angle)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    # Regular hexagon with vertices at angles 0, 60, 120, 180, 240, 300 degrees (flat-top)
    # In local frame, the hexagon is defined by |lx| <= s, |ly| <= s*sqrt(3)/2, and |lx + ly/sqrt(3)| <= s
    # Actually for a regular hexagon with side s, the six supporting lines are at distance s*sqrt(3)/2 from center
    # Let's use the standard check: the hexagon is the intersection of 6 half-planes
    # For a pointy-top hexagon (vertex at top), vertices at angles 30, 90, 150, 210, 270, 330
    # For a flat-top hexagon (edge at top), vertices at angles 0, 60, 120, 180, 240, 300
    # The seed uses angle=0, so let's match that. A regular hexagon with "angle" rotation.
    # Standard: vertices at angles angle + k*60 degrees.
    # The distance from center to each edge is s * cos(30) = s * sqrt(3)/2.
    # The six edge normals point at angles angle + 30 + k*60 degrees.
    h = s * math.sqrt(3.0) / 2.0
    for k in range(6):
        na = angle + math.pi / 6.0 + k * math.pi / 3.0
        nx = math.cos(na)
        ny = math.sin(na)
        if lx * nx + ly * ny > h:
            return False
    return True


def _poly_overlap(p1, p2):
    """Check if two convex polygons (lists of (x,y)) overlap using SAT. Returns True if they overlap (including touching)."""
    def project(pts, ax, ay):
        return min(p[0] * ax + p[1] * ay for p in pts), max(p[0] * ax + p[1] * ay for p in pts)

    polys = [p1, p2]
    for pi in range(2):
        pts = polys[pi]
        for i in range(len(pts)):
            j = (i + 1) % len(pts)
            ex = pts[j][0] - pts[i][0]
            ey = pts[j][1] - pts[i][1]
            # Normal
            nl = math.sqrt(ex * ex + ey * ey)
            if nl < 1e-12:
                continue
            nx = -ey / nl
            ny = ex / nl
            min1, max1 = project(p1, nx, ny)
            min2, max2 = project(p2, nx, ny)
            if max1 < min2 or max2 < min1:
                return False
    return True


def _outer_contains_inner(inner_pts, cx, cy, s, angle):
    """Check if all points of inner_pts are inside the outer hexagon."""
    for (px, py) in inner_pts:
        if not _point_in_hex(px, py, cx, cy, s, angle):
            return False
    return True


def _check_valid(inner, center, side, angle):
    """Validate that inner hexagons are inside outer and don't overlap each other. Returns True if valid."""
    cx, cy = center
    # Check outer containment
    for (ix, iy, ia) in inner:
        pts = _hex_points(ix, iy, 1.0, ia)
        if not _outer_contains_inner(pts, cx, cy, side, angle):
            return False
    # Check no overlap between inner hexagons
    n = len(inner)
    hex_pts = []
    for (ix, iy, ia) in inner:
        hex_pts.append(_hex_points(ix, iy, 1.0, ia))
    for i in range(n):
        for j in range(i + 1, n):
            if _poly_overlap(hex_pts[i], hex_pts[j]):
                return False
    return True


def _hex_lattice_positions(n):
    """Generate positions for n hexagons on a hexagonal lattice (side 1, touching)."""
    # Axial coordinates
    positions = []
    # Center
    positions.append((0, 0))
    # Ring r
    r = 1
    while len(positions) < n:
        # 6 directions
        dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        # Start at (r, 0)
        q, rr = r, 0
        for d in range(6):
            for _ in range(r):
                positions.append((q, rr))
                if len(positions) >= n:
                    break
                q += dirs[d][0]
                rr += dirs[d][1]
            if len(positions) >= n:
                break
        r += 1
    # Convert axial to Cartesian (pointy-top)
    # x = s * sqrt(3) * (q + r/2)
    # y = s * 3/2 * r
    # s = 1
    sqrt3 = math.sqrt(3.0)
    cart = []
    for (q, rr) in positions[:n]:
        x = sqrt3 * (q + rr * 0.5)
        y = 1.5 * rr
        cart.append((x, y, 0.0))
    return cart


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin

    # Generate hex lattice positions
    positions = _hex_lattice_positions(N)

    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # The outer hexagon can be rotated. For a hex lattice packing, the optimal outer
    # hexagon orientation is typically aligned with the lattice (angle = 0 or 30 deg).
    # Let's try a few outer angles.
    outer_angles = [0.0, math.pi / 6.0, math.pi / 3.0]

    for outer_angle in outer_angles:
        # Try to find the minimum outer side length
        # Binary search on side length
        lo = 1.0
        hi = 10.0

        # First, check if hi is valid
        def is_valid_side(s):
            return _check_valid(positions, (0.0, 0.0), s, outer_angle)

        if not is_valid_side(hi):
            hi *= 2
            while not is_valid_side(hi):
                hi *= 2
                if hi > 1000:
                    break

        if not is_valid_side(hi):
            continue

        # Binary search
        for _ in range(40):
            if time.time() > deadline - 10:
                break
            mid = (lo + hi) / 2.0
            if is_valid_side(mid):
                hi = mid
            else:
                lo = mid

        if hi < best_side:
            best_side = hi
            best_inner = positions
            best_center = (0.0, 0.0)
            best_angle = outer_angle

    # If no valid packing found, fall back to a large hexagon
    if best_inner is None:
        return positions, (0.0, 0.0), 100.0, 0.0

    # Convert to the expected format: [x, y, angle_degrees]
    inner_out = [[x, y, math.degrees(a)] for (x, y, a) in best_inner]
    return inner_out, best_center, best_side, math.degrees(best_angle)

# EVOLVE-BLOCK-END
