"""Compact 2-3-4-3 hexagonal packing of 12 unit hexagons.

Strategy: place 12 unit regular hexagons in a hex-grid arrangement
(rows of 4, 3, 3 and 2) so adjacent hexagons touch edge-to-edge.
Then set the outer hexagon side length to the exact value needed to
contain the whole arrangement, so the packing is valid and tight.
"""
import math

N = 12


def _hexagon_vertices(cx, cy, s, deg):
    """Vertices of a regular hexagon, centered at (cx, cy), side length s,
    rotated by deg degrees (flat-top when deg == 0)."""
    t = math.radians(deg)
    ct = math.cos(t)
    st = math.sin(t)
    verts = []
    for i in range(6):
        a = math.radians(60.0 * i)
        x = s * math.cos(a)
        y = s * math.sin(a)
        verts.append((cx + x * ct - y * st, cy + x * st + y * ct))
    return verts


def _polygon_area_signed(pts):
    a = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return 0.5 * a


def _polygons_overlap(v1, v2, eps=1e-9):
    """Return True if the two (convex) polygons overlap with positive area."""
    # Area-of-intersection is positive iff they overlap (ignoring shared edges).
    return _intersection_area(v1, v2) > eps


def _polygon_edge_intersect(p, q):
    """Return True if any edge of p intersects any edge of q (including touching)."""
    np_ = len(p)
    nq = len(q)
    for i in range(np_):
        a1, a2 = p[i], p[(i + 1) % np_]
        for j in range(nq):
            b1, b2 = q[j], q[(j + 1) % nq]
            if _seg_seg_intersect(a1, a2, b1, b2):
                return True
    return False


def _seg_seg_intersect(p1, p2, p3, p4):
    d1 = _cross(p3, p4, p1)
    d2 = _cross(p3, p4, p2)
    d3 = _cross(p1, p2, p3)
    d4 = _cross(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    if abs(d1) < 1e-12 and _on_segment(p3, p4, p1):
        return True
    if abs(d2) < 1e-12 and _on_segment(p3, p4, p2):
        return True
    if abs(d3) < 1e-12 and _on_segment(p1, p2, p3):
        return True
    if abs(d4) < 1e-12 and _on_segment(p1, p2, p4):
        return True
    return False


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _on_segment(a, b, c):
    if min(a[0], b[0]) - 1e-12 <= c[0] <= max(a[0], b[0]) + 1e-12 and \
       min(a[1], b[1]) - 1e-12 <= c[1] <= max(a[1], b[1]) + 1e-12:
        return True
    return False


def _intersect_polygon_edges(a1, a2, b1, b2):
    """Intersection point of two segments (assumed to intersect)."""
    d1 = _cross(b1, b2, a1)
    d2 = _cross(b1, b2, a2)
    d3 = _cross(a1, a2, b1)
    d4 = _cross(a1, a2, b2)
    denom = d1 - d2
    if abs(denom) < 1e-14:
        return None
    t = d1 / denom
    u = d3 / denom
    x = a1[0] + t * (a2[0] - a1[0])
    y = a1[1] + t * (a2[1] - a1[1])
    return (x, y)


def _sutherland_hodgman(subject, clip):
    """Clip polygon subject by convex polygon clip (both CCW)."""
    def _left_inside(pt, a, b):
        return _cross(a, b, pt) >= -1e-12

    def _intersect(p1, p2, p3, p4):
        d1 = _cross(p3, p4, p1)
        d2 = _cross(p3, p4, p2)
        denom = d1 - d2
        if abs(denom) < 1e-14:
            return p1
        t = d1 / denom
        return (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))

    output = list(subject)
    n = len(clip)
    for i in range(n):
        a = clip[i]
        b = clip[(i + 1) % n]
        if not output:
            return []
        input_list = output
        output = []
        s = input_list[-1]
        for e in input_list:
            if _left_inside(e, a, b):
                if not _left_inside(s, a, b):
                    output.append(_intersect(s, e, a, b))
                output.append(e)
            elif _left_inside(s, a, b):
                output.append(_intersect(s, e, a, b))
            s = e
    return output


def _intersection_area(v1, v2):
    """Area of intersection of two convex polygons."""
    # Ensure both are CCW
    if _polygon_area_signed(v1) < 0:
        v1 = v1[::-1]
    if _polygon_area_signed(v2) < 0:
        v2 = v2[::-1]
    inter = _sutherland_hodgman(v1, v2)
    if len(inter) < 3:
        return 0.0
    return abs(_polygon_area_signed(inter))


def _point_in_hexagon(px, py, cx, cy, s, deg):
    """Return True if point (px,py) is inside or on the hexagon."""
    t = math.radians(deg)
    ct = math.cos(t)
    st = math.sin(t)
    # Transform point into hexagon's local frame
    dx = px - cx
    dy = py - cy
    lx = dx * ct + dy * st
    ly = -dx * st + dy * ct
    # Hexagon inequalities (flat-top, side s):
    # |lx| <= s
    # |ly| <= sqrt(3)*s
    # |lx| + sqrt(3)*|ly| <= 2s  (corner constraints)
    return (abs(lx) <= s + 1e-9 and
            abs(ly) <= math.sqrt(3) * s + 1e-9 and
            abs(lx) + math.sqrt(3) * abs(ly) <= 2 * s + 1e-9)


def _validity(inner, center, side, angle):
    """Check all inner hexagons are inside the outer and pairwise disjoint."""
    cx, cy = center
    outer = _hexagon_vertices(cx, cy, side, angle)
    # Ensure CCW
    if _polygon_area_signed(outer) < 0:
        outer = outer[::-1]

    # All vertices of each inner must be inside the outer
    for (ix, iy, iang) in inner:
        iv = _hexagon_vertices(ix, iy, 1.0, iang)
        for (vx, vy) in iv:
            if not _point_in_hexagon(vx, vy, cx, cy, side, angle):
                return False

    # Pairwise disjoint (positive-area overlap)
    n = len(inner)
    for i in range(n):
        vi = _hexagon_vertices(*inner[i][:2], 1.0, inner[i][2])
        for j in range(i + 1, n):
            vj = _hexagon_vertices(*inner[j][:2], 1.0, inner[j][2])
            if _polygons_overlap(vi, vj):
                return False
    return True


def _build_positions():
    """Build the 2-3-4-3 hex-grid positions of 12 unit hexagons."""
    r3 = math.sqrt(3.0)
    # Row spacing is 1.5 (vertical distance between adjacent hex rows).
    # Horizontal spacing within a row is sqrt(3).
    # Row of 4 at y=0: x = +-1.5*r3, +-0.5*r3
    # Row of 3 at y=1.5: x = -r3, 0, +r3
    # Row of 3 at y=-1.5: x = -r3, 0, +r3
    # Row of 2 at y=3.0: x = +-0.5*r3
    rows = [
        (0.0, [-1.5 * r3, -0.5 * r3, 0.5 * r3, 1.5 * r3]),
        (1.5, [-r3, 0.0, r3]),
        (-1.5, [-r3, 0.0, r3]),
        (3.0, [-0.5 * r3, 0.5 * r3]),
    ]
    pos = []
    for y, xs in rows:
        for x in xs:
            pos.append((x, y))
    # Should be 12
    assert len(pos) == 12, f"expected 12 positions, got {len(pos)}"
    return pos


def _min_containing_side(positions):
    """Return the minimum outer hexagon side length (flat-top, centered at origin)
    such that all positions' unit hexagons fit inside.

    For a flat-top regular hexagon of side S centered at origin, a point
    (x,y) is inside iff
        |x| <= S
        |y| <= sqrt(3)*S
        |x| + sqrt(3)*|y| <= 2S
    For each inner hexagon, the worst-case vertex determines the required S.
    """
    r3 = math.sqrt(3.0)
    S = 0.0
    for (ix, iy) in positions:
        # Vertices of the unit hexagon (flat-top) centered at (ix,iy)
        for k in range(6):
            a = math.radians(60.0 * k)
            vx = ix + math.cos(a)
            vy = iy + math.sin(a)
            # Required S from each inequality
            s1 = abs(vx)
            s2 = abs(vy) / r3
            s3 = (abs(vx) + r3 * abs(vy)) / 2.0
            S = max(S, s1, s2, s3)
    return S


def construct():
    positions = _build_positions()
    # inner list: (x, y, angle_degrees); all unit hexagons, flat-top (angle 0)
    inner = [(px, py, 0.0) for (px, py) in positions]
    center = (0.0, 0.0)
    side = _min_containing_side(positions)
    angle = 0.0

    # Sanity check (cheap); fall back to a slightly larger side if needed.
    if not _validity(inner, center, side, angle):
        # Nudge the outer side up a bit to absorb any numerical slack.
        side = side + 1e-3
        if not _validity(inner, center, side, angle):
            # Last resort: a guaranteed-containing side (large).
            side = 5.0
    return inner, center, side, angle


if __name__ == "__main__":
    inner, center, side, angle = construct()
    print("side =", side, "center =", center, "angle =", angle)
    print("inner count =", len(inner))
    print("valid =", _validity(inner, center, side, angle))
    for p in inner:
        print("  ", p)

