"""Fresh design: structured hex packing with geometry verification + optimization."""
import math
import time
import numpy as np

N = 12
DEADLINE = time.time() + 135.0  # safety margin before 150s kill

# --- Geometry helpers ---

def hex_vertices(cx, cy, s, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy), side s, rotation angle_deg."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
    return verts

def polygon_area(v):
    """Shoelace area (absolute)."""
    n = len(v)
    area = 0.0
    for i in range(n):
        x1, y1 = v[i]
        x2, y2 = v[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0

def point_in_convex_polygon(px, py, poly):
    """True if (px,py) is inside or on boundary of convex polygon (CCW or CW)."""
    n = len(poly)
    sign = 0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if abs(cross) < 1e-9:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True

def seg_intersect(p1, p2, p3, p4):
    """True if segment p1-p2 intersects segment p3-p4 (including endpoints)."""
    def orient(a, b, c):
        v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(v) < 1e-9:
            return 0
        return 1 if v > 0 else -1
    o1 = orient(p1, p2, p3)
    o2 = orient(p1, p2, p4)
    o3 = orient(p3, p4, p1)
    o4 = orient(p3, p4, p2)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and min(p1[0], p2[0]) - 1e-9 <= p3[0] <= max(p1[0], p2[0]) + 1e-9 and \
        min(p1[1], p2[1]) - 1e-9 <= p3[1] <= max(p1[1], p2[1]) + 1e-9:
        return True
    if o2 == 0 and min(p1[0], p2[0]) - 1e-9 <= p4[0] <= max(p1[0], p2[0]) + 1e-9 and \
        min(p1[1], p2[1]) - 1e-9 <= p4[1] <= max(p1[1], p2[1]) + 1e-9:
        return True
    if o3 == 0 and min(p3[0], p4[0]) - 1e-9 <= p1[0] <= max(p3[0], p4[0]) + 1e-9 and \
        min(p3[1], p4[1]) - 1e-9 <= p1[1] <= max(p3[1], p4[1]) + 1e-9:
        return True
    if o4 == 0 and min(p3[0], p4[0]) - 1e-9 <= p2[0] <= max(p3[0], p4[0]) + 1e-9 and \
        min(p3[1], p4[1]) - 1e-9 <= p2[1] <= max(p3[1], p4[1]) + 1e-9:
        return True
    return False

def polygons_overlap_sat(A, B):
    """SAT: True if two convex polygons intersect (including touching)."""
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            # normal to edge
            dx, dy = x2 - x1, y2 - y1
            nx, ny = -dy, dx
            if nx == 0 and ny == 0:
                continue
            # project A
            minA = min(nx * px + ny * py for px, py in A)
            maxA = max(nx * px + ny * py for px, py in A)
            # project B
            minB = min(nx * px + ny * py for px, py in B)
            maxB = max(nx * px + ny * py for px, py in B)
            if maxA < minB - 1e-9 or maxB < minA - 1e-9:
                return False  # separating axis found
    return True

def check_no_overlap(inner, outer_center, outer_side, outer_angle, eps=1e-7):
    """Check all inner hexagons are inside outer and pairwise non-overlapping.
    inner: list of (x, y, angle_deg) with unit side."""
    outer = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    polys = []
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        polys.append(p)
    # inside outer
    for p in polys:
        for (px, py) in p:
            if not point_in_convex_polygon(px, py, outer):
                return False
    # pairwise non-overlap (allow touching)
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polygons_overlap_sat(polys[i], polys[j]):
                return False
    return True

def min_outer_side_for(inner, outer_center, outer_angle):
    """Minimum outer side length such that all unit hexagons fit inside.
    We compute the smallest S such that the hexagon (center, S, angle) contains all vertices.
    For a regular hexagon, a point p is inside iff its support in each of the 6 normal
    directions is <= S (since a regular hexagon of side S has inradius S*sqrt(3)/2 and
    the distance from center to each face along the normal = S*sqrt(3)/2).
    Actually for a regular hexagon of side s, the apothem (inradius) = s * sqrt(3)/2.
    A point is inside iff for each of the 6 outward normals n_k, (p - c)·n_k <= apothem.
    So we need S >= 2/sqrt(3) * max_k (p - c)·n_k for all p, i.e.
    S >= (2/sqrt(3)) * max over all vertices and all k of (p-c)·n_k.
    """
    cx, cy = outer_center
    a = math.radians(outer_angle)
    normals = []
    for i in range(6):
        # outward normal to edge i: direction of (i+0.5)*60 degrees
        theta = a + (i + 0.5) * math.pi / 3.0
        normals.append((math.cos(theta), math.sin(theta)))
    max_proj = 0.0
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        for (px, py) in p:
            dx, dy = px - cx, py - cy
            for (nx, ny) in normals:
                proj = dx * nx + dy * ny
                if proj > max_proj:
                    max_proj = proj
    return (2.0 / math.sqrt(3.0)) * max_proj

# --- Structured packing: 3 concentric rings (1 + 6 + 5) ---
# Ring 0: center at (0,0), angle 0
# Ring 1: 6 hexagons around center at radius r1, angles 0
# Ring 2: 5 hexagons at radius r2, angles 0
# All same orientation (angle 0) for simplicity; we can rotate the whole ring.

def build_structured(r1, r2, ring2_start_angle_deg, ring1_angle_deg, ring2_angle_deg, center_angle_deg=0.0):
    inner = []
    # center
    inner.append((0.0, 0.0, 0.0))
    # ring 1: 6 around
    for k in range(6):
        theta = math.radians(ring1_angle_deg + k * 60.0)
        x = r1 * math.cos(theta)
        y = r1 * math.sin(theta)
        inner.append((x, y, 0.0))
    # ring 2: 5 around, starting at ring2_start_angle_deg, spaced 72 deg
    for k in range(5):
        theta = math.radians(ring2_start_angle_deg + k * 72.0)
        x = r2 * math.cos(theta)
        y = r2 * math.sin(theta)
        inner.append((x, y, 0.0))
    return inner, (0.0, 0.0), 0.0, 0.0

def evaluate_structured(r1, r2, ring2_start, ring1_rot, ring2_rot):
    """Return (side, valid) for the structured packing."""
    inner = []
    inner.append((0.0, 0.0, 0.0))
    for k in range(6):
        theta = math.radians(ring1_rot + k * 60.0)
        inner.append((r1 * math.cos(theta), r1 * math.sin(theta), 0.0))
    for k in range(5):
        theta = math.radians(ring2_start + k * 72.0)
        inner.append((r2 * math.cos(theta), r2 * math.sin(theta), 0.0))
    outer_c = (0.0, 0.0)
    outer_ang = 0.0
    side = min_outer_side_for(inner, outer_c, outer_ang)
    if check_no_overlap(inner, outer_c, side, outer_ang):
        return side, True
    return side, False

def optimize():
    best_side = 40.0
    best_params = (2.0, 4.0, 0.0, 0.0, 0.0)
    best_valid = False

    # Coarse grid search
    r1_vals = np.linspace(1.5, 3.0, 12)
    r2_vals = np.linspace(3.0, 5.5, 12)
    start_vals = np.linspace(0, 72, 8)
    rot1_vals = [0.0]
    rot2_vals = [0.0]

    for r1 in r1_vals:
        for r2 in r2_vals:
            if r2 < r1 + 0.5:
                continue
            for s in start_vals:
                for rot1 in rot1_vals:
                    side, valid = evaluate_structured(r1, r2, s, rot1, 0.0)
                    if valid and side < best_side:
                        best_side = side
                        best_params = (r1, r2, s, rot1, 0.0)
                        best_valid = True
                    if time.time() > DEADLINE:
                        return best_side, best_params, best_valid

    # Local refinement around best
    if best_valid:
        r1, r2, s, rot1, rot2 = best_params
        # Nelder-Mead-like refinement via simple coordinate descent
        for it in range(200):
            if time.time() > DEADLINE:
                break
            improved = False
            for delta_r1 in (-0.05, 0.05):
                nr1 = r1 + delta_r1
                if nr1 < 1.0:
                    continue
                side, valid = evaluate_structured(nr1, r2, s, rot1, rot2)
                if valid and side < best_side:
                    best_side = side
                    best_params = (nr1, r2, s, rot1, rot2)
                    r1 = nr1
                    improved = True
            for delta_r2 in (-0.05, 0.05):
                nr2 = r2 + delta_r2
                if nr2 < r1 + 0.5:
                    continue
                side, valid = evaluate_structured(r1, nr2, s, rot1, rot2)
                if valid and side < best_side:
                    best_side = side
                    best_params = (r1, nr2, s, rot1, rot2)
                    r2 = nr2
                    improved = True
            for delta_s in (-5.0, 5.0):
                ns = s + delta_s
                side, valid = evaluate_structured(r1, r2, ns, rot1, rot2)
                if valid and side < best_side:
                    best_side = side
                    best_params = (r1, r2, ns, rot1, rot2)
                    s = ns
                    improved = True
            if not improved:
                break
    return best_side, best_params, best_valid

def construct():
    t0 = time.time()
    # Try structured packing optimization
    try:
        side, params, valid = optimize()
        if valid:
            r1, r2, s, rot1, rot2 = params
            inner = []
            inner.append((0.0, 0.0, 0.0))
            for k in range(6):
                theta = math.radians(rot1 + k * 60.0)
                inner.append((r1 * math.cos(theta), r1 * math.sin(theta), 0.0))
            for k in range(5):
                theta = math.radians(s + k * 72.0)
                inner.append((r2 * math.cos(theta), r2 * math.sin(theta), 0.0))
            return inner, (0.0, 0.0), side, 0.0
    except Exception:
        pass

    # Fallback: linear arrangement (always valid, side ~ 40)
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0
