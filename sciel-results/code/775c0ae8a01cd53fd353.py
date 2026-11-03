import math
import time

N = 11
START = time.time()
DEADLINE = 150.0 - 5.0  # safety margin

# Unit regular hexagon vertices (circumradius = 1), flat-top
def _unit_hex_verts():
    return [(math.cos(math.radians(a)), math.sin(math.radians(a))) for a in range(0, 360, 60)]

UNIT_V = _unit_hex_verts()

# Compact 1-2-3-2-1 cluster of unit hexagons.
# Hexagons of side 1 have width 2 (x) and height sqrt(3) (y) when flat-top.
# For a honeycomb, horizontal neighbor offset = 2, row offset = sqrt(3),
# and within a row, offset = 1 in x and sqrt(3)/2 in y.
def _build_cluster():
    s3 = math.sqrt(3)
    pts = []
    # rows of sizes 1,2,3,2,1 centered at x=0
    row_sizes = [1, 2, 3, 2, 1]
    y = 0.0
    for i, size in enumerate(row_sizes):
        # center this row at x=0
        # positions in row: for j in range(size): x = (j - (size-1)/2) * 2
        for j in range(size):
            x = (j - (size - 1) / 2.0) * 2.0
            pts.append((x, y))
        y += s3
    # shift so centroid is at origin
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    pts = [(x - cx, y - cy) for (x, y) in pts]
    return pts

CLUSTER = _build_cluster()
assert len(CLUSTER) == N

def _rot(v, ang):
    c, s = math.cos(ang), math.sin(ang)
    return (v[0]*c - v[1]*s, v[0]*s + v[1]*c)

def _hex_verts(cx, cy, ang):
    return [(_rot(v, ang)[0] + cx, _rot(v, ang)[1] + cy) for v in UNIT_V]

def _point_in_hex(px, py, cx, cy, ang):
    # transform point to hex local frame
    dx = px - cx
    dy = py - cy
    c, s = math.cos(ang), math.sin(ang)
    lx = dx*c + dy*s
    ly = -dx*s + dy*c
    # regular hexagon with vertices at angles 0,60,...: |x|<=1 and |x|/2 + sqrt(3)/2|y| <= 1
    return abs(lx) <= 1.0 + 1e-9 and (abs(lx)/2.0 + (math.sqrt(3)/2.0)*abs(ly)) <= 1.0 + 1e-9

def _segment_intersects(a, b, c, d):
    # orientation
    def orient(p, q, r):
        v = (q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])
        if v > 1e-12: return 1
        if v < -1e-12: return -1
        return 0
    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    if o1 != o2 and o3 != o4:
        return True
    return False

def _separating_axis(a, b):
    # a, b are lists of 6 vertices each (closed polygon as 6 pts)
    # check all edges of both polygons
    polys = [a, b]
    for poly in polys:
        for i in range(6):
            p1 = poly[i]
            p2 = poly[(i+1) % 6]
            # edge vector
            ex = p2[0] - p1[0]
            ey = p2[1] - p1[1]
            # normal
            nx, ny = -ey, ex
            # project both
            min_a = max_a = p1[0]*nx + p1[1]*ny
            for p in a:
                d = p[0]*nx + p[1]*ny
                if d < min_a: min_a = d
                if d > max_a: max_a = d
            min_b = max_b = p1[0]*nx + p1[1]*ny
            for p in b:
                d = p[0]*nx + p[1]*ny
                if d < min_b: min_b = d
                if d > max_b: max_b = d
            if max_a < min_b - 1e-9 or max_b < min_a - 1e-9:
                return True  # separating axis found
    return False

def _inner_overlap(i, j, ang):
    # check two unit hexagons at CLUSTER[i], CLUSTER[j] with rotation ang
    vi = _hex_verts(CLUSTER[i][0], CLUSTER[i][1], ang)
    vj = _hex_verts(CLUSTER[j][0], CLUSTER[j][1], ang)
    return _separating_axis(vi, vj) is False  # False means no separating axis => overlap

def _inner_cluster_valid(ang):
    # all pairs must be disjoint
    n = len(CLUSTER)
    for i in range(n):
        for j in range(i+1, n):
            if not _separating_axis(_hex_verts(CLUSTER[i][0], CLUSTER[i][1], ang),
                                    _hex_verts(CLUSTER[j][0], CLUSTER[j][1], ang)):
                return False
    return True

def _outer_contains_inner(cx, cy, side, outer_ang, inner_ang):
    # outer hexagon: center (cx,cy), side length 'side', rotation outer_ang
    # inner hexagons: CLUSTER points, rotation inner_ang
    # check each inner vertex is inside outer, and no edge crossing
    outer_v = []
    for v in UNIT_V:
        rv = _rot(v, outer_ang)
        outer_v.append((rv[0]*side + cx, rv[1]*side + cy))
    for (ix, iy) in CLUSTER:
        iv = _hex_verts(ix, iy, inner_ang)
        # each inner vertex inside outer
        for p in iv:
            if not _point_in_hex(p[0], p[1], cx, cy, outer_ang):
                return False
        # edge crossings
        for i in range(6):
            for j in range(6):
                if _segment_intersects(iv[i], iv[(i+1)%6], outer_v[j], outer_v[(j+1)%6]):
                    return False
    return True

def _min_side_for_placement(cx, cy, outer_ang, inner_ang):
    # find minimal side length such that outer contains all inner
    # binary search on side
    lo = 0.0
    hi = 50.0
    # ensure hi is valid
    for _ in range(3):
        if _outer_contains_inner(cx, cy, hi, outer_ang, inner_ang):
            break
        hi *= 2
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if _outer_contains_inner(cx, cy, mid, outer_ang, inner_ang):
            hi = mid
        else:
            lo = mid
    return hi

def construct():
    global START, DEADLINE
    best_side = 1e9
    best_params = (0.0, 0.0, 0.0, 0.0)  # cx, cy, outer_ang, inner_ang

    # The cluster is fixed in shape; we can rotate it (inner_ang) and the outer (outer_ang).
    # By symmetry, we can fix inner_ang = 0 and search outer_ang in [0, 30) degrees
    # and translate outer center. But translation of outer relative to cluster:
    # Since cluster is symmetric, we can try outer center = cluster centroid (0,0)
    # and search outer_ang. Also try small translations.

    # Coarse grid search over outer rotation and center
    # The cluster bounding: width ~ 4, height ~ 3*sqrt(3) ~ 5.196
    # Minimal outer side likely around 3-4.

    # Search outer_ang in [0, 30) deg (hex symmetry), center in small grid
    time_limit = DEADLINE
    step_ang = 2.0  # degrees
    cx_range = (-1.0, 1.0)
    cy_range = (-1.0, 1.0)
    cx_step = 0.25
    cy_step = 0.25

    # First, verify cluster is valid at inner_ang=0
    if not _inner_cluster_valid(0.0):
        # try small rotations
        for ia in [math.radians(a) for a in range(0, 30, 2)]:
            if _inner_cluster_valid(ia):
                break
        else:
            return [[x, y, 0.0] for (x, y) in CLUSTER], (0.0, 0.0), 40.0, 0.0

    inner_ang = 0.0

    # Coarse search
    ang = 0.0
    found_any = False
    while ang < math.radians(30.0):
        if time.time() > time_limit - 10:
            break
        for cx in [round(v, 2) for v in [cx_range[0] + i*cx_step for i in range(int((cx_range[1]-cx_range[0])/cx_step)+1)]]:
            for cy in [round(v, 2) for v in [cy_range[0] + i*cy_step for i in range(int((cy_range[1]-cy_range[0])/cy_step)+1)]]:
                if time.time() > time_limit - 10:
                    break
                side = _min_side_for_placement(cx, cy, ang, inner_ang)
                if side < best_side:
                    best_side = side
                    best_params = (cx, cy, ang, inner_ang)
                    found_any = True
        ang += math.radians(step_ang)

    # If nothing found, fallback
    if not found_any:
        return [[x, y, 0.0] for (x, y) in CLUSTER], (0.0, 0.0), 40.0, 0.0

    # Refine around best
    cx0, cy0, ang0, ia0 = best_params
    # refine angles
    for _ in range(3):
        if time.time() > time_limit - 3:
            break
        # ternary-like: try nearby
        best_local = best_side
        best_local_params = best_params
        for da in [math.radians(d) for d in [-1, -0.5, -0.2, 0.2, 0.5, 1]]:
            for dcx in [-0.1, 0, 0.1]:
                for dcy in [-0.1, 0, 0.1]:
                    if time.time() > time_limit - 3:
                        break
                    side = _min_side_for_placement(cx0+dcx, cy0+dcy, ang0+da, ia0)
                    if side < best_local:
                        best_local = side
                        best_local_params = (cx0+dcx, cy0+dcy, ang0+da, ia0)
        best_side = best_local
        best_params = best_local_params
        cx0, cy0, ang0, ia0 = best_params

    cx, cy, outer_ang, inner_ang = best_params
    # final side
    side = _min_side_for_placement(cx, cy, outer_ang, inner_ang)

    inner = [[x, y, math.degrees(inner_ang)] for (x, y) in CLUSTER]
    return inner, (cx, cy), side, math.degrees(outer_ang)
