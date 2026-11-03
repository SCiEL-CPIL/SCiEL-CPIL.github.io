"""
Hexagon packing: 11 unit hexagons inside a regular hexagon.
Rigid-body cluster optimization with high-precision outer hexagon polish.
"""
import math
import time
import itertools
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_SQRT3 = math.sqrt(3)
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # safety margin within 150s limit


def _hex_vertices(cx, cy, angle_deg=0.0):
    """Get vertices of a unit regular hexagon centered at (cx, cy) with rotation."""
    angle_rad = math.radians(angle_deg)
    angles = angle_rad + np.arange(6) * math.pi / 3
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])


def _compute_all_vertices(inner_hexes):
    """Precompute all vertices of inner hexagons as a numpy array (N*6, 2)."""
    n_inner = len(inner_hexes)
    verts = np.zeros((n_inner * 6, 2))
    for idx, (ix, iy, iangle) in enumerate(inner_hexes):
        verts[idx * 6: idx * 6 + 6] = _hex_vertices(ix, iy, iangle)
    return verts


def _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg):
    """For a fixed outer hexagon angle, find optimal center and side length using LP.
    Returns (cx, cy, side) or None on failure.
    """
    n_pts = all_vertices.shape[0]
    if n_pts == 0:
        return (0.0, 0.0, 0.0)

    # Normals to the 6 sides of the outer hexagon
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])  # (6, 2)

    # Vectorized construction
    v_dot_n = all_vertices @ normals.T  # (n_pts, 6)

    A_rows = []
    b_vals = []

    for k in range(6):
        nx, ny = normals[k]
        v_dots = v_dot_n[:, k]

        # Row 1: -nx*cx - ny*cy - R <= -v_dot
        A_rows.append(np.column_stack([np.full(n_pts, -nx), np.full(n_pts, -ny), np.full(n_pts, -1.0)]))
        b_vals.append(-v_dots)

        # Row 2: nx*cx + ny*cy - R <= v_dot
        A_rows.append(np.column_stack([np.full(n_pts, nx), np.full(n_pts, ny), np.full(n_pts, -1.0)]))
        b_vals.append(v_dots)

    A = np.vstack(A_rows)
    b = np.concatenate(b_vals)

    # R >= 0
    A = np.vstack([A, [[0.0, 0.0, -1.0]]])
    b = np.append(b, 0.0)

    c_obj = [0.0, 0.0, 1.0]
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b,
                      bounds=[(None, None), (None, None), (0, None)],
                      method='highs')
        if res.success:
            cx, cy, R = res.x
            side = R / (_SQRT3 / 2)
            return (cx, cy, side)
    except Exception:
        pass

    # Fallback: use centroid and compute required R
    cx_mean = all_vertices[:, 0].mean()
    cy_mean = all_vertices[:, 1].mean()
    centered = all_vertices - np.array([cx_mean, cy_mean])
    projs = centered @ normals.T  # (n_pts, 6)
    R_opt = projs.max()
    if R_opt < 0:
        R_opt = 0.0
    side = R_opt / (_SQRT3 / 2)
    return (cx_mean, cy_mean, side)


def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon."""
    dx = px - cx
    dy = py - cy
    a = -math.radians(angle_deg)
    lx = dx * math.cos(a) - dy * math.sin(a)
    ly = dx * math.sin(a) + dy * math.cos(a)
    apothem = side * _SQRT3 / 2
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
        v1a = (verts1[i][0], verts1[i][1])
        v1b = (verts1[(i + 1) % 6][0], verts1[(i + 1) % 6][1])
        for j in range(6):
            v2a = (verts2[j][0], verts2[j][1])
            v2b = (verts2[(j + 1) % 6][0], verts2[(j + 1) % 6][1])
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


def _build_positions(layout_idx, scale=1.0000001):
    """Generate 11 positions for given layout index using hexagonal lattice."""
    u = (_SQRT3 * scale, 0.0)
    v = (_SQRT3 / 2 * scale, 1.5 * scale)

    def axial_to_xy(q, r):
        x = 1.5 * q * scale
        y = _SQRT3 * (r + q / 2.0) * scale
        return (x, y)

    if layout_idx == 0:
        # Standard hexagonal: center + ring1 + 4 from ring2
        pos_axial = [(0, 0)]
        ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
        pos_axial.extend(ring1)
        # Pick 4 from ring2 that form a compact cluster
        ring2 = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
                 (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
                 (0, 2), (1, 2)]
        # Choose 4 that are close together
        selected = [ring2[0], ring2[1], ring2[2], ring2[3]]  # (2,0),(2,-1),(2,-2),(1,-2)
        pos_axial.extend(selected)
    elif layout_idx == 1:
        # Alternative compact cluster
        pos_axial = [(0, 0)]
        ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
        pos_axial.extend(ring1)
        ring2 = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
                 (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
                 (0, 2), (1, 2)]
        selected = [ring2[0], ring2[1], ring2[12], ring2[13]]  # (2,0),(2,-1),(0,2),(1,2)
        pos_axial.extend(selected)
    elif layout_idx == 2:
        # Another compact arrangement
        pos_axial = [(0, 0)]
        ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
        pos_axial.extend(ring1)
        ring2 = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
                 (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
                 (0, 2), (1, 2)]
        selected = [ring2[5], ring2[6], ring2[7], ring2[8]]  # (-1,-2),(-2,-2),(-2,-1),(-2,0)
        pos_axial.extend(selected)
    else:
        # More diverse layouts
        pos_axial = [(0, 0)]
        ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
        pos_axial.extend(ring1)
        ring2 = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
                 (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
                 (0, 2), (1, 2)]
        idxs = [layout_idx * 3 % 14, (layout_idx * 3 + 1) % 14, 
                (layout_idx * 3 + 2) % 14, (layout_idx * 3 + 3) % 14]
        selected = [ring2[i] for i in idxs]
        pos_axial.extend(selected)

    return [axial_to_xy(q, r) for q, r in pos_axial]


def _rotate_and_translate(positions, angle_deg, tx, ty):
    """Rotate positions around origin and translate."""
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    rotated = []
    for x, y in positions:
        rx = x * cos_a - y * sin_a + tx
        ry = x * sin_a + y * cos_a + ty
        rotated.append((rx, ry))
    return rotated


def _min_enclosing_hex_side(inner_hexes):
    """Find minimum side length of a regular hexagon that contains all inner hexagons.
    Returns (center_x, center_y, side_length, angle_deg)."""
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0

    all_vertices = _compute_all_vertices(inner_hexes)

    # First pass: coarse sweep
    for angle_deg in np.arange(0, 60, 2.0):
        res = _min_enclosing_hex_side_lp(all_vertices, float(angle_deg))
        if res:
            cx, cy, side = res
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)

    # Second pass: fine sweep
    if best_side < float('inf'):
        for angle_deg in np.arange(max(0.0, best_angle - 5), min(60.0, best_angle + 5), 0.25):
            res = _min_enclosing_hex_side_lp(all_vertices, float(angle_deg))
            if res:
                cx, cy, side = res
                if side < best_side:
                    best_side = side
                    best_center = (cx, cy)
                    best_angle = float(angle_deg)

    # Third pass: even finer sweep
    if best_side < float('inf'):
        for angle_deg in np.arange(max(0.0, best_angle - 1.0), min(60.0, best_angle + 1.0), 0.05):
            res = _min_enclosing_hex_side_lp(all_vertices, float(angle_deg))
            if res:
                cx, cy, side = res
                if side < best_side:
                    best_side = side
                    best_center = (cx, cy)
                    best_angle = float(angle_deg)

    # Fourth pass: Nelder-Mead polish
    if best_side < float('inf') and time.time() < _DEADLINE - 15.0:
        def objective(params):
            ang_p = params[2] % 60.0
            res = _min_enclosing_hex_side_lp(all_vertices, float(ang_p))
            if res:
                return res[2]
            return 1e6

        x0 = np.array([best_center[0], best_center[1], best_angle])
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12,
                                   'adaptive': True})
            if res.success:
                ang_nm = float(res.x[2] % 60.0)
                cx_v, cy_v, side_v = _min_enclosing_hex_side_lp(all_vertices, ang_nm)
                if side_v < best_side:
                    best_side = side_v
                    best_center = (cx_v, cy_v)
                    best_angle = ang_nm
        except Exception:
            pass

    # Fifth pass: High-precision grid search
    if best_side < float('inf') and time.time() < _DEADLINE - 5.0:
        window = 0.1  # degrees
        steps = 200
        angles_hp = np.linspace(best_angle - window, best_angle + window, steps)
        for ang in angles_hp:
            ang_mod = ang % 60.0
            if ang_mod < 0:
                ang_mod += 60.0
            res = _min_enclosing_hex_side_lp(all_vertices, float(ang_mod))
            if res:
                cx, cy, side = res
                if side < best_side:
                    best_side = side
                    best_center = (cx, cy)
                    best_angle = float(ang_mod)

    return best_center[0], best_center[1], best_side, best_angle


def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_center = (0.0, 0.0)
    best_oangle = 0.0

    # Try multiple rigid-body layouts with various global rotations
    n_layouts = 10
    rotation_angles = [0.0, 15.0, 30.0, 45.0, -15.0, -30.0, -45.0]

    for layout_idx in range(n_layouts):
        if time.time() > deadline:
            break

        base_positions = _build_positions(layout_idx)

        for rot_angle in rotation_angles:
            if time.time() > deadline:
                break

            # Rotate the cluster
            rotated_positions = _rotate_and_translate(base_positions, rot_angle, 0.0, 0.0)

            # All inner hexagons have the same rotation as the cluster
            inner_hexes = [(x, y, 1.0, rot_angle) for x, y in rotated_positions]

            # Find minimum enclosing hexagon
            center_x, center_y, side, oangle = _min_enclosing_hex_side(inner_hexes)

            if side < best_side:
                best_side = side
                best_positions = rotated_positions[:]
                best_angles = [rot_angle] * len(rotated_positions)
                best_center = (center_x, center_y)
                best_oangle = oangle

    if best_positions is None:
        # Fallback: simple lattice
        base_positions = _build_positions(0)
        best_positions = base_positions
        best_angles = [0.0] * 11
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in base_positions]
        best_center, best_side = _min_enclosing_hex_side(inner_hexes)[:2]
        best_oangle = 0.0

    # Verify disjointness and containment
    inner_hexes = [(best_positions[i][0], best_positions[i][1], 1.0, best_angles[i]) for i in range(len(best_positions))]

    if not _all_disjoint(inner_hexes):
        # Scale up positions slightly to ensure disjointness
        for scale in [1.001, 1.005, 1.01, 1.02]:
            scaled_hexes = [(x * scale, y * scale, 1.0, a) for x, y, s, a in inner_hexes]
            if _all_disjoint(scaled_hexes):
                inner_hexes = scaled_hexes
                best_positions = [(h[0], h[1]) for h in inner_hexes]
                best_angles = [h[3] for h in inner_hexes]
                center_x, center_y, side, oangle = _min_enclosing_hex_side(inner_hexes)
                best_side = side
                best_center = (center_x, center_y)
                best_oangle = oangle
                break

    # Verify containment
    outer_hex = (best_center[0], best_center[1], best_side, best_oangle)
    valid = True
    for h in inner_hexes:
        if not _inner_in_outer(h, outer_hex):
            valid = False
            break

    if not valid:
        # Increase side until valid
        for _ in range(100):
            best_side *= 1.001
            outer_hex = (best_center[0], best_center[1], best_side, best_oangle)
            if all(_inner_in_outer(h, outer_hex) for h in inner_hexes):
                break

    # Format output
    inner = [[inner_hexes[i][0], inner_hexes[i][1], inner_hexes[i][3]] for i in range(len(inner_hexes))]

    return inner, (best_center[0], best_center[1]), best_side, best_oangle
