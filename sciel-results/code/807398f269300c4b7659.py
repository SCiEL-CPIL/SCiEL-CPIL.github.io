"""
Hexagon packing: 11 unit hexagons inside a regular hexagon.
Combined approach:
- Structural: lattice enumeration over C(12,4) second-ring topologies (from current program)
- Refinement: multi-resolution outer-angle sweep with exact LP (from inspiration)
- Smooth surrogate for joint position/angle/outer optimization (log-sum-exp)
- Nelder-Mead polish on outer angle using exact LP
- Safety: final validation and scaling
"""
import math
import time
import itertools
import numpy as np
from scipy.optimize import minimize, linprog

N = 11
_DEADLINE = time.time() + 150.0 - 15.0
EPS = 1e-9
SQRT3 = math.sqrt(3)


def _hex_vertices(cx, cy, side, angle_deg):
    verts = []
    for k in range(6):
        a = math.radians(angle_deg + 60 * k)
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return verts


def _point_in_hex(px, py, cx, cy, side, angle_deg):
    dx = px - cx
    dy = py - cy
    a = -math.radians(angle_deg)
    lx = dx * math.cos(a) - dy * math.sin(a)
    ly = dx * math.sin(a) + dy * math.cos(a)
    apothem = side * SQRT3 / 2
    for k in range(6):
        a_norm = math.radians(30 + 60 * k)
        nx = math.cos(a_norm)
        ny = math.sin(a_norm)
        if lx * nx + ly * ny > apothem + 1e-9:
            return False
    return True


def _segments_intersect(p1, p2, p3, p4):
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
    cx1, cy1, s1, a1 = hex1
    cx2, cy2, s2, a2 = hex2
    dx = cx2 - cx1
    dy = cy2 - cy1
    if dx * dx + dy * dy > (s1 + s2) ** 2 + 1e-12:
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


def _all_disjoint(inner_hexes):
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            if not _hexagons_disjoint(inner_hexes[i], inner_hexes[j]):
                return False
    return True


def _inner_in_outer(inner_hex, outer_hex):
    cx, cy, s, a = inner_hex
    ox, oy, oside, oangle = outer_hex
    verts = _hex_vertices(cx, cy, s, a)
    for vx, vy in verts:
        if not _point_in_hex(vx, vy, ox, oy, oside, oangle):
            return False
    return True


def _compute_outer_lp(positions, angles, oangle):
    """Exact LP for minimum enclosing hexagon side. Returns (cx, cy, side)."""
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
        side = res.x[2] * 2.0 / SQRT3
    else:
        cx_mean = sum(p[0] for p in positions) / len(positions)
        cy_mean = sum(p[1] for p in positions) / len(positions)
        center = (cx_mean, cy_mean)
        side = float('inf')
    return center, side


def _search_outer_angle(positions, angles, coarse=1.0, fine=0.2, finer=0.05, finest=0.01):
    """Multi-resolution sweep over outer angle using exact LP."""
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0

    for ang in np.arange(0, 60, coarse):
        if time.time() > _DEADLINE - 8:
            break
        center, side = _compute_outer_lp(positions, angles, float(ang))
        if side < best_side:
            best_side = side
            best_center = center
            best_angle = float(ang)

    if best_side == float('inf'):
        return best_center, best_side, best_angle

    for ang in np.arange(max(0, best_angle - 1.0), min(60, best_angle + 1.0), fine):
        if time.time() > _DEADLINE - 8:
            break
        center, side = _compute_outer_lp(positions, angles, float(ang))
        if side < best_side:
            best_side = side
            best_center = center
            best_angle = float(ang)

    for ang in np.arange(max(0, best_angle - 0.3), min(60, best_angle + 0.3), finer):
        if time.time() > _DEADLINE - 8:
            break
        center, side = _compute_outer_lp(positions, angles, float(ang))
        if side < best_side:
            best_side = side
            best_center = center
            best_angle = float(ang)

    for ang in np.arange(max(0, best_angle - 0.1), min(60, best_angle + 0.1), finest):
        if time.time() > _DEADLINE - 8:
            break
        center, side = _compute_outer_lp(positions, angles, float(ang))
        if side < best_side:
            best_side = side
            best_center = center
            best_angle = float(ang)

    return best_center, best_side, best_angle


def _polish_outer_angle(positions, angles, ang0):
    """Golden-section search on outer angle using exact LP."""
    if time.time() > _DEADLINE - 12:
        return None

    def obj(ang):
        ang = float(ang % 60.0)
        _, side = _compute_outer_lp(positions, angles, ang)
        return side

    try:
        a, b = max(0.0, ang0 - 0.05), min(60.0, ang0 + 0.05)
        gr = (math.sqrt(5.0) - 1.0) / 2.0
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = obj(c)
        fd = obj(d)
        for _ in range(60):
            if time.time() > _DEADLINE - 10:
                break
            if fc < fd:
                b, d = d, c
                fd = fc
                c = b - gr * (b - a)
                fc = obj(c)
            else:
                a, c = c, d
                fc = fd
                d = a + gr * (b - a)
                fd = obj(d)
        best_ang = (a + b) / 2.0
        center, side = _compute_outer_lp(positions, angles, best_ang)
        return center, side, best_ang
    except Exception:
        return None


def _support_function(cx, cy, s, ang, nx, ny):
    max_val = 0.0
    for m in range(6):
        a = math.radians(ang + 60 * m)
        vx = s * math.cos(a)
        vy = s * math.sin(a)
        val = vx * nx + vy * ny
        if val > max_val:
            max_val = val
    return max_val


def _pairwise_violation(state, angles_list, i, j):
    max_violation = 0.0
    for k in range(12):
        theta = math.radians(15 * k)
        nx = math.cos(theta)
        ny = math.sin(theta)
        s_i = _support_function(state[2 * i], state[2 * i + 1], 1.0, angles_list[i], nx, ny)
        s_j = _support_function(state[2 * j], state[2 * j + 1], 1.0, angles_list[j], -nx, -ny)
        dx = state[2 * j] - state[2 * i]
        dy = state[2 * j + 1] - state[2 * i + 1]
        dist_proj = dx * nx + dy * ny
        violation = s_i + s_j - dist_proj
        if violation > max_violation:
            max_violation = violation
    return max_violation


def _smooth_outer_side(state, n_hex, oangle, alpha=30.0):
    """Smooth log-sum-exp surrogate for outer hexagon side."""
    vals = np.empty(n_hex * 6 * 6)
    idx = 0
    for i in range(n_hex):
        cx = state[3 * i]
        cy = state[3 * i + 1]
        ang = state[3 * i + 2]
        for k in range(6):
            a = math.radians(ang + 60 * k)
            vx = cx + math.cos(a)
            vy = cy + math.sin(a)
            for m in range(6):
                nx = math.cos(math.radians(oangle + 30 + 60 * m))
                ny = math.sin(math.radians(oangle + 30 + 60 * m))
                vals[idx] = vx * nx + vy * ny
                idx += 1
    m = np.max(vals)
    t_smooth = m + np.log(np.sum(np.exp(vals - m)) / len(vals)) / alpha
    side = t_smooth * 2.0 / SQRT3
    return side


def _smooth_penalty(state, n_hex, beta=50.0):
    penalty = 0.0
    angles_list = [state[3 * m + 2] for m in range(n_hex)]
    for i in range(n_hex):
        for j in range(i + 1, n_hex):
            v = _pairwise_violation(state, angles_list, i, j)
            if v > 0:
                penalty += v ** 2
    return penalty


def construct():
    start_time = time.time()

    def axial_to_xy(q, r):
        x = 1.5 * q
        y = SQRT3 * (r + q / 2.0)
        return (x, y)

    # Hexagonal lattice: center + ring1 (6) + ring2 (12) = 19 positions
    pos_axial = [(0, 0)]
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    pos_axial.extend(ring1)

    ring2 = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
             (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
             (0, 2), (1, 2)]

    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_center = (0.0, 0.0)
    best_oangle = 0.0

    # Phase 1: Lattice enumeration over C(12,4) second-ring topologies
    # With 7 base positions (center + ring1) we need 4 from ring2
    # We try multiple outer angles per topology
    oangles_phase1 = [0.0, 15.0, 30.0, -15.0, -30.0, 45.0, -45.0, 10.0, -10.0, 20.0, -20.0]

    combo_count = 0
    total_combos = len(list(itertools.combinations(range(12), 4)))
    for combo in itertools.combinations(range(12), 4):
        if time.time() > _DEADLINE - 60:
            break
        combo_count += 1
        selected_r2 = [ring2[i] for i in combo]
        full_axial = pos_axial + selected_r2
        full_xy = [axial_to_xy(q, r) for q, r in full_axial]

        for oangle in oangles_phase1:
            if time.time() > _DEADLINE - 60:
                break
            center, side = _compute_outer_lp(full_xy, [0.0] * 11, oangle)
            if side < best_side:
                best_side = side
                best_positions = full_xy[:]
                best_angles = [0.0] * 11
                best_center = center
                best_oangle = oangle

    if best_positions is None:
        full_xy = [axial_to_xy(q, r) for q, r in pos_axial[:7]]
        best_positions = full_xy
        best_angles = [0.0] * 7
        best_center, best_side = _compute_outer_lp(best_positions, best_angles, 0.0)
        best_oangle = 0.0

    # Phase 2: Multi-resolution outer angle refinement on best topology
    if time.time() < _DEADLINE - 30:
        best_center, best_side, best_oangle = _search_outer_angle(
            best_positions, best_angles, coarse=1.0, fine=0.2, finer=0.05, finest=0.01)

    # Phase 3: Golden-section polish on outer angle
    if time.time() < _DEADLINE - 20:
        result = _polish_outer_angle(best_positions, best_angles, best_oangle)
        if result is not None:
            cx, cy, side, ang = result
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_oangle = ang

    # Phase 4: Smooth L-BFGS-B joint optimization (positions + angles)
    if time.time() < _DEADLINE - 25:
        n_hex = len(best_positions)

        def objective(state):
            side = _smooth_outer_side(state, n_hex, best_oangle, alpha=30.0)
            penalty = _smooth_penalty(state, n_hex, beta=50.0)
            return side + 100.0 * penalty

        def gradient(state):
            grad = np.zeros_like(state)
            h = 1e-7
            f0 = objective(state)
            for i in range(len(state)):
                state[i] += h
                f1 = objective(state)
                state[i] -= h
                grad[i] = (f1 - f0) / h
            return grad

        init_state = []
        for i in range(n_hex):
            init_state.extend([best_positions[i][0], best_positions[i][1], best_angles[i]])
        init_state = np.array(init_state)

        try:
            res = minimize(objective, init_state, method='L-BFGS-B',
                           jac=gradient,
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < best_side + 50:
                final_state = res.x
                hexes = [(final_state[3 * i], final_state[3 * i + 1], 1.0, final_state[3 * i + 2])
                         for i in range(n_hex)]
                if _all_disjoint(hexes):
                    positions = [(h[0], h[1]) for h in hexes]
                    angles = [h[3] for h in hexes]
                    # Re-optimize outer angle for new positions
                    center, side = _search_outer_angle(positions, angles,
                                                       coarse=2.0, fine=0.5,
                                                       finer=0.1, finest=0.02)
                    if side < best_side:
                        best_side = side
                        best_positions = positions
                        best_angles = angles
                        best_center = center
                        best_oangle = center[2] if len(center) > 2 else best_oangle
                        # _search_outer_angle returns (center, side, angle)
                        # Fix: unpack properly
                        best_center, best_side, best_oangle = _search_outer_angle(
                            best_positions, best_angles,
                            coarse=2.0, fine=0.5, finer=0.1, finest=0.02)
        except Exception:
            pass

    # Phase 5: High-precision polish
    if time.time() < _DEADLINE - 10:
        n_hex = len(best_positions)

        def polish_objective(state):
            side = _smooth_outer_side(state, n_hex, best_oangle, alpha=50.0)
            penalty = _smooth_penalty(state, n_hex, beta=100.0)
            return side + 200.0 * penalty

        def polish_gradient(state):
            grad = np.zeros_like(state)
            h = 1e-8
            f0 = polish_objective(state)
            for i in range(len(state)):
                state[i] += h
                f1 = polish_objective(state)
                state[i] -= h
                grad[i] = (f1 - f0) / h
            return grad

        init_state = []
        for i in range(n_hex):
            init_state.extend([best_positions[i][0], best_positions[i][1], best_angles[i]])
        init_state = np.array(init_state)

        try:
            res = minimize(polish_objective, init_state, method='L-BFGS-B',
                           jac=polish_gradient,
                           options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < best_side + 10:
                final_state = res.x
                hexes = [(final_state[3 * i], final_state[3 * i + 1], 1.0, final_state[3 * i + 2])
                         for i in range(n_hex)]
                if _all_disjoint(hexes):
                    positions = [(h[0], h[1]) for h in hexes]
                    angles = [h[3] for h in hexes]
                    best_center, best_side, best_oangle = _search_outer_angle(
                        positions, angles,
                        coarse=1.0, fine=0.2, finer=0.05, finest=0.01)
                    best_positions = positions
                    best_angles = angles
        except Exception:
            pass

    # Phase 6: Final golden-section polish on outer angle
    if time.time() < _DEADLINE - 5:
        result = _polish_outer_angle(best_positions, best_angles, best_oangle)
        if result is not None:
            cx, cy, side, ang = result
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_oangle = ang

    # Final verification and safety scaling
    inner_hexes = [(best_positions[i][0], best_positions[i][1], 1.0, best_angles[i])
                   for i in range(len(best_positions))]

    if not _all_disjoint(inner_hexes):
        for scale in [1.001, 1.005, 1.01, 1.02, 1.05]:
            scaled_hexes = [(x * scale, y * scale, 1.0, a) for x, y, s, a in inner_hexes]
            if _all_disjoint(scaled_hexes):
                inner_hexes = scaled_hexes
                best_positions = [(h[0], h[1]) for h in inner_hexes]
                best_angles = [h[3] for h in inner_hexes]
                best_center, best_side = _compute_outer_lp(
                    best_positions, best_angles, best_oangle)
                break

    outer_hex = (best_center[0], best_center[1], best_side, best_oangle)
    valid = True
    for h in inner_hexes:
        if not _inner_in_outer(h, outer_hex):
            valid = False
            break

    if not valid:
        for _ in range(200):
            best_side *= 1.001
            outer_hex = (best_center[0], best_center[1], best_side, best_oangle)
            if all(_inner_in_outer(h, outer_hex) for h in inner_hexes):
                break

    inner = [[inner_hexes[i][0], inner_hexes[i][1], inner_hexes[i][3]]
             for i in range(len(inner_hexes))]

    return inner, (best_center[0], best_center[1]), best_side, best_oangle
