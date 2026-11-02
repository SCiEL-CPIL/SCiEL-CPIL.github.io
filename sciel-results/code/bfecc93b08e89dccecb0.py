"""Fresh design: hex-lattice centers + brute-force minimal enclosing hexagon."""
import time
import math
import random
import numpy as np

N = 11
_DEADLINE_MARGIN = 8.0   # seconds before the 120s wall-clock limit
_START = time.time()


def _elapsed():
    return time.time() - _START


def _deadline():
    return _START + 120.0 - _DEADLINE_MARGIN


def _hex_vertices(cx, cy, s, theta):
    """Vertices of a regular hexagon (side s) centered at (cx,cy), rotation theta (rad)."""
    # Standard: vertices at angles theta + k*pi/3, k=0..5
    # For a regular hexagon of side s, the circumradius is s.
    vs = []
    for k in range(6):
        a = theta + k * math.pi / 3.0
        vs.append((cx + s * math.cos(a), cy + s * math.sin(a)))
    return vs


def _hex_vertex_array(cx, cy, s, theta):
    """Return (6,2) ndarray of vertices."""
    a = theta + np.arange(6) * (math.pi / 3.0)
    return np.column_stack((cx + s * np.cos(a), cy + s * np.sin(a)))


def _point_in_convex_polygon(p, verts):
    """Return True if point p is inside (or on) convex polygon verts (ccw order)."""
    # verts: (6,2)
    # Use cross product test
    x0, y0 = p
    ok = True
    n = len(verts)
    for i in range(n):
        ax, ay = verts[i]
        bx, by = verts[(i + 1) % n]
        # cross (b-a, p-a)
        cx = (bx - ax) * (y0 - ay) - (by - ay) * (x0 - ax)
        if cx < -1e-12:
            ok = False
            break
    return ok


def _hexagon_contains_point(cx, cy, s, theta, px, py):
    """Does the hexagon (center, side s, rotation theta) contain point (px,py)?"""
    # Transform point into hexagon's local frame
    dx = px - cx
    dy = py - cy
    ca = math.cos(-theta)
    sa = math.sin(-theta)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    # Regular hexagon centered at origin with vertices at angles 0, pi/3, ...
    # The hexagon is the intersection of 6 half-planes: |dot(p, v_i)| <= s * cos(pi/6) * s
    # Actually: inradius = s * cos(pi/6) = s * sqrt(3)/2
    # The 6 normals are at angles pi/6, pi/2, 5pi/6, 7pi/6, 3pi/2, 11pi/6
    inradius = s * math.sqrt(3) / 2.0
    for k in range(6):
        a = math.pi / 6.0 + k * math.pi / 3.0
        nx = math.cos(a)
        ny = math.sin(a)
        if nx * lx + ny * ly > inradius + 1e-12:
            return False
    return True


def _hexagon_contains_all_points(cx, cy, s, theta, pts):
    """pts: (M,2) array of points. True if all inside hexagon."""
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    ca = math.cos(-theta)
    sa = math.sin(-theta)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    inradius = s * math.sqrt(3) / 2.0
    for k in range(6):
        a = math.pi / 6.0 + k * math.pi / 3.0
        nx = math.cos(a)
        ny = math.sin(a)
        if np.max(nx * lx + ny * ly) > inradius + 1e-9:
            return False
    return True


def _min_s_for_angle(pts, cx, cy, theta):
    """Minimal side s of a hexagon centered at (cx,cy) with rotation theta containing all pts."""
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    ca = math.cos(-theta)
    sa = math.sin(-theta)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    max_proj = 0.0
    for k in range(6):
        a = math.pi / 6.0 + k * math.pi / 3.0
        nx = math.cos(a)
        ny = math.sin(a)
        m = np.max(np.abs(nx * lx + ny * ly))
        if m > max_proj:
            max_proj = m
    # inradius = s * sqrt(3)/2  =>  s = max_proj / (sqrt(3)/2) = 2*max_proj/sqrt(3)
    return 2.0 * max_proj / math.sqrt(3.0)


def _compute_min_enclosing_hex(pts, center_hint=None):
    """Brute force over rotation angle to find minimal enclosing regular hexagon.
    Returns (side, cx, cy, angle_deg)."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    if center_hint is None:
        cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
    else:
        cx, cy = center_hint

    best_s = float('inf')
    best_theta = 0.0
    # Coarse grid over angle (full symmetry: hexagon has 6-fold symmetry, so period pi/3)
    n_coarse = 36
    for i in range(n_coarse):
        theta = i * (math.pi / 3.0) / n_coarse
        s = _min_s_for_angle(pts, cx, cy, theta)
        if s < best_s:
            best_s = s
            best_theta = theta

    # Refine around best_theta
    for _ in range(3):
        dtheta = best_s * 0.001 + 1e-4
        lo = best_theta - dtheta
        hi = best_theta + dtheta
        n_fine = 40
        local_best = (best_s, best_theta)
        for i in range(n_fine + 1):
            theta = lo + (hi - lo) * i / n_fine
            s = _min_s_for_angle(pts, cx, cy, theta)
            if s < local_best[0]:
                local_best = (s, theta)
        if abs(local_best[0] - best_s) < 1e-7 and abs(local_best[1] - best_theta) < 1e-7:
            break
        best_s, best_theta = local_best

    return (best_s, cx, cy, best_theta * 180.0 / math.pi)


def _hex_vertices_from_center_side(cx, cy, s, theta_deg):
    """Return list of (x,y) vertices of inner hexagon centered at (cx,cy), side s, rotation theta_deg."""
    theta = theta_deg * math.pi / 180.0
    vs = []
    for k in range(6):
        a = theta + k * math.pi / 3.0
        vs.append((cx + s * math.cos(a), cy + s * math.sin(a)))
    return vs


def _point_in_hexagon(px, py, cx, cy, s, theta_deg):
    theta = theta_deg * math.pi / 180.0
    return _hexagon_contains_point(cx, cy, s, theta, px, py)


def _segments_intersect(p1, p2, p3, p4, tol=1e-9):
    """Proper/imp proper segment intersection test."""
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    o1 = orient(p1, p2, p3)
    o2 = orient(p1, p2, p4)
    o3 = orient(p3, p4, p1)
    o4 = orient(p3, p4, p2)
    if ((o1 > tol and o2 < -tol) or (o1 < -tol and o2 > tol)) and \
       ((o3 > tol and o4 < -tol) or (o3 < -tol and o4 > tol)):
        return True
    return False


def _polygons_disjoint(verts1, verts2, tol=1e-9):
    """Two convex polygons are disjoint if no edges cross AND neither contains a vertex of the other."""
    n1 = len(verts1)
    n2 = len(verts2)
    # Check all edge pairs
    for i in range(n1):
        a1 = verts1[i]
        a2 = verts1[(i + 1) % n1]
        for j in range(n2):
            b1 = verts2[j]
            b2 = verts2[(j + 1) % n2]
            if _segments_intersect(a1, a2, b1, b2, tol):
                return False
    # Check vertex containment
    for p in verts1:
        if _point_in_hexagon(p[0], p[1], *[_center_of_hex(verts2)]):
            return False
    for p in verts2:
        if _point_in_hexagon(p[0], p[1], *[_center_of_hex(verts1)]):
            return False
    return True


def _center_of_hex(verts):
    cx = sum(v[0] for v in verts) / len(verts)
    cy = sum(v[1] for v in verts) / len(verts)
    # side length: distance from center to a vertex
    dx = verts[0][0] - cx
    dy = verts[0][1] - cy
    s = math.sqrt(dx * dx + dy * dy)
    # rotation: angle of first vertex relative to center
    theta = math.atan2(dy, dx)
    return (cx, cy, s, theta * 180.0 / math.pi)


def _hexagons_disjoint(inner, tol=1e-8):
    """inner: list of [cx, cy, angle_deg]. Unit side hexagons. Return True if all pairwise disjoint."""
    n = len(inner)
    # Precompute vertices
    all_verts = []
    for (cx, cy, ang) in inner:
        vs = _hex_vertices(cx, cy, 1.0, ang * math.pi / 180.0)
        all_verts.append(vs)
    for i in range(n):
        for j in range(i + 1, n):
            if not _polygons_disjoint(all_verts[i], all_verts[j], tol):
                return False
    return True


def _build_hex_lattice_positions(k, spacing):
    """Return list of (x,y) positions of unit hexagons on a hex lattice that are pairwise disjoint.
    k = ring count. spacing is center-to-center distance between adjacent hexagons."""
    # Hex lattice: axial coordinates (q, r). Neighbor distance = spacing.
    # For unit hexagons (side 1), adjacent centers must be at distance >= 2 * inradius = sqrt(3)
    # to be disjoint (touching). Use spacing slightly larger for numerical safety.
    positions = []
    # Hexagonal grid of radius k (k rings around center)
    # Axial coordinates with |q| <= k, |r| <= k, |q+r| <= k
    for q in range(-k, k + 1):
        for r in range(-k, k + 1):
            if abs(q + r) > k:
                continue
            # Convert axial to cartesian (pointy-top vs flat-top)
            # Flat-top: x = spacing * (q + r/2), y = spacing * (sqrt(3)/2 * r)
            x = spacing * (q + r / 2.0)
            y = spacing * (math.sqrt(3) / 2.0 * r)
            positions.append((x, y))
    return positions


def _all_disjoint_positions(k, spacing):
    """Generate all positions on hex lattice of radius k, then greedily select a maximal
    set that is pairwise disjoint (with strict gap)."""
    positions = _build_hex_lattice_positions(k, spacing)
    # For unit hexagons, disjointness requires center distance >= 2 * inradius = sqrt(3).
    # On a hex lattice with spacing, adjacent centers are exactly spacing apart.
    # So we need spacing >= sqrt(3) + eps.
    selected = []
    min_dist = spacing  # on hex lattice, minimum distance is spacing
    # All positions are on the lattice, so any two distinct positions have distance >= spacing.
    # If spacing >= sqrt(3) + eps, all are disjoint.
    # So just return all positions up to N.
    # But we want compact (small bounding hexagon). So pick positions closest to origin.
    positions_sorted = sorted(positions, key=lambda p: p[0] * p[0] + p[1] * p[1])
    return positions_sorted


def construct():
    """Construct N disjoint unit hexagons inside a minimal regular hexagon.
    Returns (inner, (cx, cy), side, angle_deg)."""
    inner = None
    best_outer = None
    best_side = float('inf')

    # Strategy 1: hex lattice with compact set of N positions
    # spacing must be >= sqrt(3) for disjointness of unit hexagons
    spacing = math.sqrt(3) * 1.0001  # tiny epsilon for numerical safety

    # Generate candidate position sets:
    # - "spiral": take N closest positions to origin on hex lattice
    # - "ring subsets": various selections
    candidate_sets = []

    # Spiral (closest first)
    all_pos = _all_disjoint_positions(6, spacing)
    if len(all_pos) >= N:
        candidate_sets.append(all_pos[:N])

    # Try different spacings (larger spacing -> more gap but larger outer hex)
    # Smaller spacing -> tighter but might violate disjointness.
    # Actually on hex lattice, any two distinct positions are at distance >= spacing.
    # So minimum distance is exactly spacing. For disjointness we need spacing >= sqrt(3).
    # So spacing = sqrt(3) is the tightest.

    # Try also "block" arrangements: e.g., a 2x2x2 block on hex lattice
    # But for N=11, the spiral of 11 is already quite compact.

    # Evaluate each candidate set
    for positions in candidate_sets:
        if len(positions) != N:
            continue
        pts = np.array(positions, dtype=float)
        cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))

        # Compute minimal enclosing hexagon
        side, ocx, ocy, oang = _compute_min_enclosing_hex(pts, center_hint=(cx, cy))

        # Verify disjointness of inner hexagons
        inner_list = [[p[0], p[1], 0.0] for p in positions]
        if not _hexagons_disjoint(inner_list):
            continue

        # Verify containment: each inner hexagon's vertices must be inside outer hex
        contained = True
        for (ix, iy, iang) in inner_list:
            ivers = _hex_vertices(ix, iy, 1.0, iang * math.pi / 3.0)
            for (vx, vy) in ivers:
                if not _hexagon_contains_point(ocx, ocy, side, oang * math.pi / 180.0, vx, vy):
                    contained = False
                    break
            if not contained:
                break

        if not contained:
            # Expand slightly
            side *= 1.001
            contained = True
            for (ix, iy, iang) in inner_list:
                ivers = _hex_vertices(ix, iy, 1.0, iang * math.pi / 3.0)
                for (vx, vy) in ivers:
                    if not _hexagon_contains_point(ocx, ocy, side, oang * math.pi / 180.0, vx, vy):
                        contained = False
                        break
                if not contained:
                    break

        if not contained:
            continue

        if side < best_side:
            best_side = side
            inner = inner_list
            best_outer = (ocx, ocy, side, oang)

    # Strategy 2: if strategy 1 failed, try a simple line arrangement
    if inner is None:
        # Place hexagons in a line with spacing sqrt(3)
        positions = []
        for i in range(N):
            x = (i - (N - 1) / 2.0) * math.sqrt(3)
            y = 0.0
            positions.append((x, y))
        inner_list = [[p[0], p[1], 0.0] for p in positions]
        if _hexagons_disjoint(inner_list):
            pts = np.array(positions, dtype=float)
            cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
            side, ocx, ocy, oang = _compute_min_enclosing_hex(pts, center_hint=(cx, cy))
            # Verify containment
            contained = True
            for (ix, iy, iang) in inner_list:
                ivers = _hex_vertices(ix, iy, 1.0, iang * math.pi / 3.0)
                for (vx, vy) in ivers:
                    if not _hexagon_contains_point(ocx, ocy, side, oang * math.pi / 180.0, vx, vy):
                        contained = False
                        break
                if not contained:
                    break
            if not contained:
                side *= 1.001
            inner = inner_list
            best_outer = (ocx, ocy, side, oang)
            best_side = side

    # Fallback: large hexagon
    if inner is None:
        positions = []
        for i in range(N):
            x = (i - (N - 1) / 2.0) * (math.sqrt(3) * 1.01)
            y = 0.0
            positions.append((x, y))
        inner = [[p[0], p[1], 0.0] for p in positions]
        best_outer = (0.0, 0.0, 50.0, 0.0)
        best_side = 50.0

    # Final verification
    if best_outer is None:
        return inner, (0.0, 0.0), 50.0, 0.0

    ocx, ocy, side, oang = best_outer

    # Verify once more
    if not _hexagons_disjoint(inner):
        return inner, (0.0, 0.0), 50.0, 0.0

    contained = True
    for (ix, iy, iang) in inner:
        ivers = _hex_vertices(ix, iy, 1.0, iang * math.pi / 3.0)
        for (vx, vy) in ivers:
            if not _hexagon_contains_point(ocx, ocy, side, oang * math.pi / 180.0, vx, vy):
                contained = False
                break
        if not contained:
            break

    if not contained:
        side *= 1.01
        contained = True
        for (ix, iy, iang) in inner:
            ivers = _hex_vertices(ix, iy, 1.0, iang * math.pi / 3.0)
            for (vx, vy) in ivers:
                if not _hexagon_contains_point(ocx, ocy, side, oang * math.pi / 180.0, vx, vy):
                    contained = False
                    break
            if not contained:
                break

    if not contained:
        return inner, (0.0, 0.0), 50.0, 0.0

    return inner, (ocx, ocy), side, oang


# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
