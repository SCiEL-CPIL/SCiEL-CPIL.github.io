"""
Combined approach: structural lattice enumeration (inspiration) + continuous
joint optimization with LP-free exact outer-hexagon evaluation (current program
ideas).  The key structural change is to stop assuming the outer hexagon is
centered at the origin and instead compute the *exact* minimum enclosing regular
hexagon for any fixed set of inner hexagons by optimizing its center and angle.
We then jointly optimize inner positions and outer parameters.
"""
import math
import time
import itertools
import numpy as np
from scipy.optimize import minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # Safety margin before 150 s limit


def _hex_vertices(cx, cy, side=1.0, angle_deg=0.0):
    """6 vertices of a regular hexagon (side length `side`) centered at (cx, cy),
    rotated by angle_deg degrees.  Vertices are at angles angle_deg + 60*k."""
    verts = []
    for k in range(6):
        a = math.radians(angle_deg + 60 * k)
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return verts


def _point_in_hex(px, py, cx, cy, side=1.0, angle_deg=0.0):
    """Check if point (px, py) is inside (or on boundary of) a regular hexagon
    centered at (cx, cy) with side length `side`, rotated by angle_deg."""
    dx = px - cx
    dy = py - cy
    a = -math.radians(angle_deg)
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    rx = dx * cos_a - dy * sin_a
    ry = dx * sin_a + dy * cos_a
    apothem = side * math.sqrt(3) / 2
    for k in range(6):
        theta = math.radians(30 + 60 * k)
        nx = math.cos(theta)
        ny = math.sin(theta)
        if rx * nx + ry * ny > apothem + 1e-9:
            return False
    return True


def _segments_intersect(p1, p2, p3, p4):
    """Proper (strict) intersection of segments p1-p2 and p3-p4."""
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


def _hexagons_disjoint(h1, h2, eps=1e-9):
    """Return True if two hexagons (each a list of 6 (x,y) tuples) have disjoint
    interiors.  Touching is allowed."""
    # Quick bounding-box / distance check
    # Compute approximate center distance
    cx1 = sum(v[0] for v in h1) / 6
    cy1 = sum(v[1] for v in h1) / 6
    cx2 = sum(v[0] for v in h2) / 6
    cy2 = sum(v[1] for v in h2) / 6
    dx = cx2 - cx1
    dy = cy2 - cy1
    if dx * dx + dy * dy > 4.0 + 1e-12:  # (1+1)^2
        return True

    verts1 = h1
    verts2 = h2
    # Vertex-in-polygon checks
    for vx, vy in verts1:
        # Check against each vertex of h2 as center? No, we need the hexagon params.
        # We'll pass the hexagon params separately in the caller.
        # Actually, let's just do the check inline with the given vertices.
        # For convex polygons, if any vertex of one is strictly inside the other, they overlap.
        pass  # We'll handle this in the caller with proper params

    # Edge-edge intersection
    for i in range(6):
        v1a = verts1[i]
        v1b = verts1[(i + 1) % 6]
        for j in range(6):
            v2a = verts2[j]
            v2b = verts2[(j + 1) % 6]
            if _segments_intersect(v1a, v1b, v2a, v2b):
                return False
    return True


def _check_disjoint(positions, angles, eps=1e-9):
    """Check that all pairs of inner hexagons have disjoint interiors.
    positions: list of (x, y)
    angles: list of angle_degrees (same length as positions)
    """
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = positions[i]
            x2, y2 = positions[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx * dx + dy * dy > 4.0 + 1e-12:
                continue
            verts1 = _hex_vertices(x1, y1, 1.0, angles[i])
            verts2 = _hex_vertices(x2, y2, 1.0, angles[j])
            # Vertex-in-hex checks
            for vx, vy in verts1:
                if _point_in_hex(vx, vy, x2, y2, 1.0, angles[j]):
                    return False
            for vx, vy in verts2:
                if _point_in_hex(vx, vy, x1, y1, 1.0, angles[i]):
                    return False
            # Edge-edge intersection
            for a in range(6):
                for b in range(6):
                    if _segments_intersect(verts1[a], verts1[(a + 1) % 6],
                                           verts2[b], verts2[(b + 1) % 6]):
                        return False
    return True


def _min_outer_side(positions, angles, ox, oy, oangle):
    """Compute the minimum side length of a regular hexagon centered at (ox, oy)
    with rotation oangle that contains all inner hexagons.
    Returns the side length S."""
    max_proj = 0.0
    a = -math.radians(oangle)
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    for i in range(len(positions)):
        px, py = positions[i]
        verts = _hex_vertices(px, py, 1.0, angles[i])
        for vx, vy in verts:
            dx = vx - ox
            dy = vy - oy
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            for k in range(6):
                theta = math.radians(30 + 60 * k)
                proj = lx * math.cos(theta) + ly * math.sin(theta)
                if proj > max_proj:
                    max_proj = proj
    return max_proj * 2.0 / math.sqrt(3)


def _optimize_outer_for_fixed_inner(positions, angles):
    """Given fixed inner hexagon positions and angles, find the optimal outer
    hexagon center (ox, oy) and angle oangle that minimizes the enclosing side
    length.  Uses a small local optimization."""
    # Initial guess: centroid of all vertices
    all_verts = []
    for i in range(len(positions)):
        px, py = positions[i]
        all_verts.extend(_hex_vertices(px, py, 1.0, angles[i]))
    cx0 = sum(v[0] for v in all_verts) / len(all_verts)
    cy0 = sum(v[1] for v in all_verts) / len(all_verts)

    best_side = float('inf')
    best_params = (cx0, cy0, 0.0)

    for oang_init in [0.0, 15.0, 30.0, 45.0, 60.0, 75.0]:
        def obj_outer(x):
            ox, oy, oang = x
            return _min_outer_side(positions, angles, ox, oy, oang)

        try:
            res = minimize(obj_outer, [cx0, cy0, oang_init], method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_side:
                best_side = res.fun
                best_params = (res.x[0], res.x[1], res.x[2])
        except Exception:
            pass

    return best_side, best_params


def construct():
    global _START_TIME, _DEADLINE

    if time.time() > _DEADLINE - 5.0:
        # Emergency fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    sqrt3 = math.sqrt(3)

    # Phase 1: Lattice-based enumeration of promising configurations
    # Use axial coordinates for flat-top hexagonal lattice
    def axial_to_xy(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return (x, y)

    # Center + ring 1 (6 hexagons)
    pos_axial = [(0, 0)]
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    pos_axial.extend(ring1)

    # Ring 2 positions
    ring2 = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
             (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
             (0, 2), (1, 2)]

    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # Enumerate combinations of 4 ring-2 positions
    all_combos = list(itertools.combinations(range(len(ring2)), 4))

    for combo in all_combos:
        if time.time() > _DEADLINE - 50.0:
            break

        selected_r2 = [ring2[i] for i in combo]
        full_axial = pos_axial + selected_r2
        full_xy = [axial_to_xy(q, r) for q, r in full_axial]
        angles = [0.0] * N

        # Optimize outer hexagon for this fixed inner configuration
        side, (ox, oy, oang) = _optimize_outer_for_fixed_inner(full_xy, angles)

        if side < best_side:
            best_side = side
            best_positions = full_xy
            best_angles = angles
            best_center = (ox, oy)
            best_angle = oang

    if best_positions is None:
        # Fallback
        best_positions = [axial_to_xy(q, r) for q, r in pos_axial]
        best_angles = [0.0] * 7
        # Need 11, so add 4 more
        for i in range(4):
            best_positions.append((best_positions[i][0] + 3.0, best_positions[i][1]))
            best_angles.append(0.0)
        best_side, (ox, oy, oang) = _optimize_outer_for_fixed_inner(best_positions, best_angles)
        best_center = (ox, oy)
        best_angle = oang

    # Phase 2: Joint continuous optimization
    # Optimize inner positions (and possibly angles) to further reduce the outer side
    if time.time() < _DEADLINE - 15.0:
        # Start from the best lattice configuration
        x0 = np.array([coord for pos in best_positions for coord in pos])
        # We'll keep angles fixed at best_angle for simplicity, but allow positions to move

        def obj_joint(x):
            if time.time() > _DEADLINE - 10.0:
                return float('inf')
            positions = [(x[i * 2], x[i * 2 + 1]) for i in range(N)]
            angles = [best_angle] * N
            # Check disjointness (penalty if violated)
            if not _check_disjoint(positions, angles):
                # Estimate penalty
                penalty = 0.0
                for i in range(N):
                    for j in range(i + 1, N):
                        dx = positions[j][0] - positions[i][0]
                        dy = positions[j][1] - positions[i][1]
                        d = math.hypot(dx, dy)
                        if d < 2.0:
                            penalty += (2.0 - d) ** 2
                return 100.0 + penalty
            side, (ox, oy, oang) = _optimize_outer_for_fixed_inner(positions, angles)
            return side

        try:
            res = minimize(obj_joint, x0, method='L-BFGS-B',
                           options={'maxiter': 100, 'ftol': 1e-8})
            if res.fun < best_side and res.fun < 100.0:
                positions_res = [(res.x[i * 2], res.x[i * 2 + 1]) for i in range(N)]
                angles_res = [best_angle] * N
                if _check_disjoint(positions_res, angles_res):
                    side_res, (ox_res, oy_res, oang_res) = _optimize_outer_for_fixed_inner(positions_res, angles_res)
                    if side_res < best_side:
                        best_side = side_res
                        best_positions = positions_res
                        best_angles = angles_res
                        best_center = (ox_res, oy_res)
                        best_angle = oang_res
        except Exception:
            pass

    # Phase 3: Fine-tune outer hexagon
    if time.time() < _DEADLINE - 5.0:
        side_final, (ox_f, oy_f, oang_f) = _optimize_outer_for_fixed_inner(best_positions, best_angles)
        if side_final < best_side:
            best_side = side_final
            best_center = (ox_f, oy_f)
            best_angle = oang_f

    # Build output
    inner = [[best_positions[i][0], best_positions[i][1], best_angles[i]] for i in range(N)]
    return inner, best_center, best_side, best_angle
