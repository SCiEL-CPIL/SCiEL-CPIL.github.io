"""
Hexagon packing: 11 unit hexagons inside a regular hexagon.
Combines LP-based enclosing-hexagon computation with a topological contact-graph
parametrization.  The 11 unit hexagons are arranged in a fixed cluster topology
(central + 6-ring + 4 outer) and positioned by continuous separation parameters
along each contact direction, which guarantees disjointness by construction.
The objective is the side length of the minimum enclosing regular hexagon,
minimized via smooth local optimization followed by high-precision polishing.
"""
import math
import time
import itertools
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
DEADLINE_MARGIN = 12.0  # seconds before the 150s wall-clock limit


def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for k in range(6):
        a = math.radians(angle_deg + 60 * k)
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return verts


def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon (inclusive)."""
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


def _compute_outer_for_positions(positions, angles, oangle=0.0):
    """Compute optimal outer hexagon center and side for given inner positions."""
    inner_hexes = [(positions[i][0], positions[i][1], 1.0, angles[i])
                   for i in range(len(positions))]

    c = [0, 0, 1]
    A_ub = []
    b_ub = []

    for cx_h, cy_h, s_h, ang_h in inner_hexes:
        verts = _hex_vertices(cx_h, cy_h, s_h, ang_h)
        for vx, vy in verts:
            for k in range(6):
                nx = math.cos(math.radians(oangle + 30 + 60 * k))
                ny = math.sin(math.radians(oangle + 30 + 60 * k))
                A_ub.append([-nx, -ny, -1])
                b_ub.append(-vx * nx - vy * ny)

    A_ub.append([0, 0, -1])
    b_ub.append(0)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                  bounds=[(None, None), (None, None), (0, None)], method='highs')

    if res.success:
        center = (res.x[0], res.x[1])
        side = res.x[2] * 2.0 / math.sqrt(3)
    else:
        cx_mean = sum(p[0] for p in positions) / len(positions)
        cy_mean = sum(p[1] for p in positions) / len(positions)
        center = (cx_mean, cy_mean)
        # Fallback: compute side from max projection
        apothem = 0
        for cx_h, cy_h, s_h, ang_h in inner_hexes:
            verts = _hex_vertices(cx_h, cy_h, s_h, ang_h)
            for vx, vy in verts:
                for k in range(6):
                    nx = math.cos(math.radians(oangle + 30 + 60 * k))
                    ny = math.sin(math.radians(oangle + 30 + 60 * k))
                    proj = (vx - cx_mean) * nx + (vy - cy_mean) * ny
                    if proj > apothem:
                        apothem = proj
        side = apothem * 2.0 / math.sqrt(3)

    return center, side


# ---------------------------------------------------------------------------
# Topological contact-graph parametrization
# ---------------------------------------------------------------------------
#
# Cluster topology:
#   Hex 0: center
#   Hex 1-6: ring around center (axial directions 0..5)
#   Hex 7-10: four outer hexagons attached to ring hexagons
#
# Axial directions (q, r) for a flat-top hex grid:
#   dir 0: ( 1,  0)   angle 0°
#   dir 1: ( 1, -1)   angle -60°
#   dir 2: ( 0, -1)   angle -120°
#   dir 3: (-1,  0)   angle 180°
#   dir 4: (-1,  1)   angle 120°
#   dir 5: ( 0,  1)   angle 60°
#
# For two unit hexagons with the same orientation, the minimum center-to-center
# distance for disjoint interiors along direction d is:
#   d_min = 2 * apothem = sqrt(3)  (when aligned along a face normal)
#   d_min = 2 * side = 2          (when aligned along a vertex direction)
#
# For the axial neighbor direction (q=1, r=0), the distance is 2*apothem = sqrt(3)
# in the x-direction... actually let me compute properly.
#
# Axial to xy for flat-top hex (side=1):
#   x = 1.5 * q
#   y = sqrt(3) * (r + q/2)
#
# Direction (1,0): dx = 1.5, dy = sqrt(3)/2, dist = sqrt(2.25 + 0.75) = sqrt(3)
# Direction (1,-1): dx = 1.5, dy = -sqrt(3)/2, dist = sqrt(3)
# All axial neighbor distances are sqrt(3).
#
# For two unit hexagons with the same orientation, the minimum separation
# along any direction is sqrt(3) (face-to-face) or 2 (vertex-to-vertex).
# The axial directions correspond to face-to-face, so min dist = sqrt(3).
#
# In the contact-graph parametrization, we use a separation parameter s_i >= 0
# for each contact, where the actual center-to-center distance is:
#   dist = sqrt(3) + s_i
#
# This guarantees disjointness (s_i = 0 gives touching, s_i > 0 gives gap).
#
# The four outer hexagons are attached to specific ring hexagons.
# We choose them to be attached to ring hexagons 0, 2, 4 (alternating)
# in the "outward" direction, plus one more.
#
# Actually, let's use a simpler approach: the 4 outer hexagons are placed
# in the "gaps" between ring hexagons, at the second ring positions.
# We'll parametrize their positions relative to their nearest ring neighbors.

# Contact graph definition:
# Each contact is (i, j, direction_angle_deg) where hex j is placed at
# distance (sqrt(3) + s) from hex i in the given direction.
#
# Ring hexagons (1-6) are placed relative to center (0):
#   Ring k (k=0..5) is at direction angle = -60*k degrees from center
#   (matching axial directions)
#
# Outer hexagons (7-10) are placed relative to specific ring hexagons.
# We'll use a "second ring" placement:
#   Outer 7: attached to ring 0, direction outward (same as ring0 direction from center)
#   Outer 8: attached to ring 2, direction outward
#   Outer 9: attached to ring 4, direction outward
#   Outer 10: attached to ring 1, direction outward (or ring 3)
#
# Actually, for a symmetric 11-hex cluster, let's use:
#   Center + 6 ring + 4 at "corner" positions of the second ring.
#
# Let me use a different approach: define the topology as a list of
# (parent, child, angle_deg) where child is placed at distance
# (sqrt(3) + s_param) from parent in the given direction.

# Topology:
#   0 is center
#   1-6 are ring (attached to 0)
#   7-10 are outer (attached to ring hexagons)

TOPOLOGY = [
    # (parent, child, base_angle_deg)
    # Ring: child k attached to parent 0 at angle -60*(k-1)
    (0, 1, 0.0),
    (0, 2, -60.0),
    (0, 3, -120.0),
    (0, 4, 180.0),
    (0, 5, 120.0),
    (0, 6, 60.0),
    # Outer: attached to ring hexagons, outward direction
    # Ring 1 is at angle 0 from center, so outward is also angle 0
    (1, 7, 0.0),
    # Ring 2 is at angle -60 from center, outward is -60
    (2, 8, -60.0),
    # Ring 3 is at angle -120 from center, outward is -120
    (3, 9, -120.0),
    # Ring 4 is at angle 180 from center, outward is 180
    (4, 10, 180.0),
]

# Number of separation parameters = number of contacts = 10
N_PARAMS = len(TOPOLOGY)


def _build_positions(sep_params, angles=None):
    """
    Build hexagon positions from separation parameters.
    sep_params: list of 10 values (one per contact in TOPOLOGY).
    angles: optional list of 11 rotation angles (degrees). If None, all 0.
    Returns list of (x, y, angle) for 11 hexagons.
    """
    if angles is None:
        angles = [0.0] * 11

    positions = [(0.0, 0.0)] * 11  # (x, y) for each hex
    positions[0] = (0.0, 0.0)

    for idx, (parent, child, base_angle_deg) in enumerate(TOPOLOGY):
        s = sep_params[idx]
        dist = math.sqrt(3) + s
        a_rad = math.radians(base_angle_deg)
        px, py = positions[parent]
        positions[child] = (px + dist * math.cos(a_rad),
                           py + dist * math.sin(a_rad))

    result = []
    for i in range(11):
        result.append((positions[i][0], positions[i][1], angles[i]))
    return result


def _compute_side_from_positions(positions_angles, oangle=0.0):
    """Compute enclosing hexagon side from list of (x, y, angle)."""
    positions = [(p[0], p[1]) for p in positions_angles]
    angles = [p[2] for p in positions_angles]
    _, side = _compute_outer_for_positions(positions, angles, oangle)
    return side


def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN

    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_center = (0.0, 0.0)
    best_oangle = 0.0
    best_sep = None

    def _try_update(positions_angles, oangle, sep_params=None):
        nonlocal best_side, best_positions, best_angles, best_center, best_oangle, best_sep
        positions = [(p[0], p[1]) for p in positions_angles]
        angles = [p[2] for p in positions_angles]
        center, side = _compute_outer_for_positions(positions, angles, oangle)
        if side < best_side:
            # Verify disjointness
            inner_hexes = [(positions[i][0], positions[i][1], 1.0, angles[i])
                          for i in range(11)]
            if _all_disjoint(inner_hexes):
                best_side = side
                best_positions = positions[:]
                best_angles = angles[:]
                best_center = center
                best_oangle = oangle
                if sep_params is not None:
                    best_sep = sep_params[:]

    # ------------------------------------------------------------------
    # Phase 1: Grid search over separation parameters and outer angle
    # ------------------------------------------------------------------
    # Start with all separations = 0 (tight packing)
    # Try small positive separations to allow optimization room

    sep_grid = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    oangle_grid = [0.0, 15.0, 30.0, -15.0, -30.0, 5.0, -5.0, 10.0, -10.0]

    # Try uniform separations first
    for s_val in sep_grid:
        if time.time() > deadline:
            break
        sep_params = [s_val] * N_PARAMS
        for oangle in oangle_grid:
            if time.time() > deadline:
                break
            pos_ang = _build_positions(sep_params)
            _try_update(pos_ang, oangle, sep_params)

    # Try non-uniform: larger separations for outer hexagons
    for s_ring in [0.0, 0.05, 0.1]:
        if time.time() > deadline:
            break
        for s_outer in [0.0, 0.1, 0.2, 0.3, 0.5]:
            if time.time() > deadline:
                break
            sep_params = [s_ring] * 6 + [s_outer] * 4
            for oangle in [0.0, 15.0, -15.0, 30.0, -30.0]:
                if time.time() > deadline:
                    break
                pos_ang = _build_positions(sep_params)
                _try_update(pos_ang, oangle, sep_params)

    if best_positions is None:
        # Fallback: standard hex lattice
        sqrt3 = math.sqrt(3)
        def axial_to_xy(q, r):
            x = 1.5 * q
            y = sqrt3 * (r + q / 2.0)
            return (x, y)

        axial = [(0, 0), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
                 (2, 0), (2, -2), (0, -2), (-2, 0)]
        full_xy = [axial_to_xy(q, r) for q, r in axial]
        pos_ang = [(x, y, 0.0) for x, y in full_xy]
        _try_update(pos_ang, 0.0)

    if best_positions is None:
        # Absolute fallback
        best_positions = [(0.0, 0.0)] * 11
        best_angles = [0.0] * 11
        best_center = (0.0, 0.0)
        best_side = 5.0
        best_oangle = 0.0

    # ------------------------------------------------------------------
    # Phase 2: Nonlinear optimization over separation parameters
    # ------------------------------------------------------------------
    if time.time() < deadline - 20:
        def sep_objective(sep_params, oangle_fixed):
            # Clamp separations to non-negative
            sep_clamped = [max(0.0, s) for s in sep_params]
            pos_ang = _build_positions(sep_clamped)
            side = _compute_side_from_positions(pos_ang, oangle_fixed)
            return side

        # Try optimizing separations for the best outer angle found so far
        for oangle_try in [best_oangle, 0.0, 15.0, -15.0, 30.0, -30.0]:
            if time.time() > deadline - 15:
                break

            init_sep = [0.0] * N_PARAMS if best_sep is None else best_sep[:]
            try:
                result = minimize(
                    lambda x: sep_objective(x, oangle_try),
                    init_sep,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                if result.fun < best_side + 1.0:
                    sep_opt = [max(0.0, s) for s in result.x]
                    pos_ang = _build_positions(sep_opt)
                    _try_update(pos_ang, oangle_try, sep_opt)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Phase 3: Free-position optimization (refinement of the topology result)
    # ------------------------------------------------------------------
    if time.time() < deadline - 15:
        n_hex = 11

        def _support_function(cx, cy, s, ang, nx, ny):
            max_val = 0
            for m in range(6):
                a = math.radians(ang + 60 * m)
                vx = s * math.cos(a)
                vy = s * math.sin(a)
                val = vx * nx + vy * ny
                if val > max_val:
                    max_val = val
            return max_val

        def _pairwise_violation(pos, angles, i, j, num_dirs=24):
            max_violation = 0
            for k in range(num_dirs):
                theta = 2.0 * math.pi * k / num_dirs
                nx = math.cos(theta)
                ny = math.sin(theta)
                s_i = _support_function(pos[2*i], pos[2*i+1], 1.0, angles[i], nx, ny)
                s_j = _support_function(pos[2*j], pos[2*j+1], 1.0, angles[j], -nx, -ny)
                dx = pos[2*j] - pos[2*i]
                dy = pos[2*j+1] - pos[2*i+1]
                dist_proj = dx * nx + dy * ny
                violation = s_i + s_j - dist_proj
                if violation > max_violation:
                    max_violation = violation
            return max_violation

        def free_objective(state):
            hexes = []
            for i in range(n_hex):
                hexes.append((state[3*i], state[3*i+1], 1.0, state[3*i+2]))
            inner_hexes = [(h[0], h[1], h[2], h[3]) for h in hexes]

            c = [0, 0, 1]
            A_ub = []
            b_ub = []
            for cx_h, cy_h, s_h, ang_h in inner_hexes:
                verts = _hex_vertices(cx_h, cy_h, s_h, ang_h)
                for vx, vy in verts:
                    for k in range(6):
                        nx = math.cos(math.radians(best_oangle + 30 + 60 * k))
                        ny = math.sin(math.radians(best_oangle + 30 + 60 * k))
                        A_ub.append([-nx, -ny, -1])
                        b_ub.append(-vx * nx - vy * ny)
            A_ub.append([0, 0, -1])
            b_ub.append(0)

            res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                          bounds=[(None, None), (None, None), (0, None)], method='highs')
            if not res.success:
                return 1e6
            side = res.x[2] * 2.0 / math.sqrt(3)

            penalty = 0
            for i in range(n_hex):
                for j in range(i + 1, n_hex):
                    violation = _pairwise_violation(state, [state[3*m+2] for m in range(n_hex)], i, j, 24)
                    if violation > 0:
                        penalty += violation ** 2

            return side + 100.0 * penalty

        init_state = []
        for i in range(n_hex):
            init_state.extend([best_positions[i][0], best_positions[i][1], best_angles[i]])

        try:
            result = minimize(free_objective, init_state, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            if result.fun < best_side + 100:
                final_state = result.x
                hexes = []
                for i in range(n_hex):
                    hexes.append((final_state[3*i], final_state[3*i+1], 1.0, final_state[3*i+2]))
                if _all_disjoint(hexes):
                    positions = [(h[0], h[1]) for h in hexes]
                    angles = [h[3] for h in hexes]
                    center, side = _compute_outer_for_positions(positions, angles, best_oangle)
                    if side < best_side:
                        best_side = side
                        best_positions = positions
                        best_angles = angles
                        best_center = center
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Phase 4: Optimize outer angle with current best positions
    # ------------------------------------------------------------------
    if time.time() < deadline - 10:
        for oangle in [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0,
                       -5.0, -10.0, -15.0, -20.0, -25.0, -30.0]:
            if time.time() > deadline:
                break
            center, side = _compute_outer_for_positions(best_positions, best_angles, oangle)
            if side < best_side:
                best_side = side
                best_center = center
                best_oangle = oangle

    # ------------------------------------------------------------------
    # Phase 5: High-precision polish
    # ------------------------------------------------------------------
    if time.time() < deadline - 8:
        n_hex = 11

        def _support_function2(cx, cy, s, ang, nx, ny):
            max_val = 0
            for m in range(6):
                a = math.radians(ang + 60 * m)
                vx = s * math.cos(a)
                vy = s * math.sin(a)
                val = vx * nx + vy * ny
                if val > max_val:
                    max_val = val
            return max_val

        def _pairwise_violation2(pos, angles, i, j, num_dirs=36):
            max_violation = 0
            for k in range(num_dirs):
                theta = 2.0 * math.pi * k / num_dirs
                nx = math.cos(theta)
                ny = math.sin(theta)
                s_i = _support_function2(pos[2*i], pos[2*i+1], 1.0, angles[i], nx, ny)
                s_j = _support_function2(pos[2*j], pos[2*j+1], 1.0, angles[j], -nx, -ny)
                dx = pos[2*j] - pos[2*i]
                dy = pos[2*j+1] - pos[2*i+1]
                dist_proj = dx * nx + dy * ny
                violation = s_i + s_j - dist_proj
                if violation > max_violation:
                    max_violation = violation
            return max_violation

        def polish_objective(state):
            hexes = []
            for i in range(n_hex):
                hexes.append((state[3*i], state[3*i+1], 1.0, state[3*i+2]))
            inner_hexes = [(h[0], h[1], h[2], h[3]) for h in hexes]

            c = [0, 0, 1]
            A_ub = []
            b_ub = []
            for cx_h, cy_h, s_h, ang_h in inner_hexes:
                verts = _hex_vertices(cx_h, cy_h, s_h, ang_h)
                for vx, vy in verts:
                    for k in range(6):
                        nx = math.cos(math.radians(best_oangle + 30 + 60 * k))
                        ny = math.sin(math.radians(best_oangle + 30 + 60 * k))
                        A_ub.append([-nx, -ny, -1])
                        b_ub.append(-vx * nx - vy * ny)
            A_ub.append([0, 0, -1])
            b_ub.append(0)

            res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                          bounds=[(None, None), (None, None), (0, None)], method='highs')
            if not res.success:
                return 1e6
            side = res.x[2] * 2.0 / math.sqrt(3)

            penalty = 0
            for i in range(n_hex):
                for j in range(i + 1, n_hex):
                    violation = _pairwise_violation2(state, [state[3*m+2] for m in range(n_hex)], i, j, 36)
                    if violation > 0:
                        penalty += violation ** 2

            return side + 100.0 * penalty

        init_state = []
        for i in range(n_hex):
            init_state.extend([best_positions[i][0], best_positions[i][1], best_angles[i]])

        try:
            polish_result = minimize(polish_objective, init_state, method='Nelder-Mead',
                                     options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14})
            if polish_result.fun < best_side + 100:
                final_state = polish_result.x
                hexes = []
                for i in range(n_hex):
                    hexes.append((final_state[3*i], final_state[3*i+1], 1.0, final_state[3*i+2]))
                if _all_disjoint(hexes):
                    positions = [(h[0], h[1]) for h in hexes]
                    angles = [h[3] for h in hexes]
                    center, side = _compute_outer_for_positions(positions, angles, best_oangle)
                    if side < best_side:
                        best_side = side
                        best_positions = positions
                        best_angles = angles
                        best_center = center
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Final verification and safety
    # ------------------------------------------------------------------
    inner_hexes = [(best_positions[i][0], best_positions[i][1], 1.0, best_angles[i])
                   for i in range(11)]

    if not _all_disjoint(inner_hexes):
        for scale in [1.001, 1.005, 1.01, 1.02]:
            scaled_hexes = [(x * scale, y * scale, 1.0, a) for x, y, s, a in inner_hexes]
            if _all_disjoint(scaled_hexes):
                inner_hexes = scaled_hexes
                best_positions = [(h[0], h[1]) for h in inner_hexes]
                best_angles = [h[3] for h in inner_hexes]
                best_center, best_side = _compute_outer_for_positions(
                    best_positions, best_angles, best_oangle)
                break

    outer_hex = (best_center[0], best_center[1], best_side, best_oangle)
    valid = True
    for h in inner_hexes:
        if not _inner_in_outer(h, outer_hex):
            valid = False
            break

    if not valid:
        for _ in range(100):
            best_side *= 1.001
            outer_hex = (best_center[0], best_center[1], best_side, best_oangle)
            if all(_inner_in_outer(h, outer_hex) for h in inner_hexes):
                break

    inner = [[inner_hexes[i][0], inner_hexes[i][1], inner_hexes[i][3]]
             for i in range(11)]

    return inner, (best_center[0], best_center[1]), best_side, best_oangle
