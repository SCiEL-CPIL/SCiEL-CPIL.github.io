"""Smooth-surrogate outer-hexagon packing for 11 unit hexagons."""
import math
import time
import numpy as np
import itertools

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 143.0
_SQRT3 = math.sqrt(3)

def _hex_vertices_array(cx, cy, angle_deg):
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _orient(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val > 0:
        return 1
    if val < 0:
        return 2
    return 0

def _on_segment(p, q, r):
    if min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and \
       min(p[1], r[1]) <= q[1] <= max(p[1], r[1]):
        return True
    return False

def _segments_intersect(p1, p2, p3, p4):
    o1 = _orient(p1, p2, p3)
    o2 = _orient(p1, p2, p4)
    o3 = _orient(p3, p4, p1)
    o4 = _orient(p3, p4, p2)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and _on_segment(p1, p3, p2):
        return True
    if o2 == 0 and _on_segment(p1, p4, p2):
        return True
    if o3 == 0 and _on_segment(p3, p1, p4):
        return True
    if o4 == 0 and _on_segment(p3, p2, p4):
        return True

    return False

def _point_in_polygon(p, poly):
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        if ((poly[i][1] > p[1]) != (poly[j][1] > p[1])) and \
           (p[0] < (poly[j][0] - poly[i][0]) * (p[1] - poly[i][1]) / (poly[j][1] - poly[i][1]) + poly[i][0]):
            inside = not inside
        j = i
    return inside

def _polygons_intersect(v1, v2):
    # Check vertices
    for p in v1:
        if _point_in_polygon(p, v2):
            return True
    for p in v2:
        if _point_in_polygon(p, v1):
            return True
    # Check edges
    n1 = len(v1)
    n2 = len(v2)
    for i in range(n1):
        p1 = v1[i]
        p2 = v1[(i + 1) % n1]
        for j in range(n2):
            p3 = v2[j]
            p4 = v2[(j + 1) % n2]
            if _segments_intersect(p1, p2, p3, p4):
                return True
    return False

def _all_disjoint(inner_hexes):
    verts_list = []
    for h in inner_hexes:
        x, y, ang = h
        verts = _hex_vertices_array(x, y, ang)
        verts_list.append(verts)
    
    for i in range(len(verts_list)):
        for j in range(i + 1, len(verts_list)):
            if _polygons_intersect(verts_list[i], verts_list[j]):
                return False
    return True

def _compute_all_vertices(inner_hexes):
    n_inner = len(inner_hexes)
    verts = np.zeros((n_inner * 6, 2))
    for idx, (ix, iy, iangle) in enumerate(inner_hexes):
        angle_rad = math.radians(iangle)
        for k in range(6):
            a = angle_rad + math.pi * k / 3
            verts[idx * 6 + k, 0] = ix + math.cos(a)
            verts[idx * 6 + k, 1] = iy + math.sin(a)
    return verts

def _min_enclosing_hex_side_lp(verts, outer_angle_deg):
    """Exact LP: minimum R such that all points fit in hexagon with given orientation.
    Returns (cx, cy, side) or None."""
    M = len(verts)
    if M == 0:
        return 0.0, 0.0, 0.0

    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])

    v_dot_n = verts @ normals.T
    max_dots = np.max(v_dot_n, axis=0)
    min_dots = np.min(v_dot_n, axis=0)

    A_rows = []
    b_vals = []
    for k in range(6):
        nx, ny = normals[k]
        A_rows.append([-nx, -ny, -1.0])
        b_vals.append(-max_dots[k])
        A_rows.append([nx, ny, -1.0])
        b_vals.append(min_dots[k])

    A = np.array(A_rows)
    b = np.array(b_vals)
    A = np.vstack([A, [[0.0, 0.0, -1.0]]])
    b = np.append(b, 0.0)

    try:
        from scipy.optimize import linprog
        res = linprog([0.0, 0.0, 1.0], A_ub=A, b_ub=b,
                      bounds=[(None, None), (None, None), (0, None)],
                      method='highs')
        if res.success:
            cx, cy, R = res.x
            side = R / (_SQRT3 / 2)
            return cx, cy, side
    except Exception:
        pass
    return None

def _smooth_enclosing_hex(verts, outer_angle_deg, beta=80.0):
    """Smooth surrogate: minimize soft-max of support violations + R penalty.
    Returns (cx, cy, side) or None."""
    M = len(verts)
    if M == 0:
        return 0.0, 0.0, 0.0

    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])  # (6,2)
    v_dot_n = verts @ normals.T  # (M,6)

    def objective(params):
        cx, cy, R = params
        if R < 0:
            R = 0.0
        c_dot_n = cx * normals[:, 0] + cy * normals[:, 1]  # (6,)
        # For each normal k: max_v(v.n_k) - c.n_k <= R  =>  violation = max_v(v.n_k) - c.n_k - R
        # Use smooth max via log-sum-exp over vertices for each normal
        # diffs[m,k] = v_dot_n[m,k] - c_dot_n[k]
        diffs = v_dot_n - c_dot_n[np.newaxis, :]  # (M,6)
        # smooth_max per column: (1/beta)*log(sum(exp(beta*diffs)))
        max_per_col = np.max(diffs, axis=0)  # (6,)
        shifted = beta * (diffs - max_per_col[np.newaxis, :])
        # Clip for numerical stability
        shifted = np.clip(shifted, -50.0, 50.0)
        log_sum = np.log(np.sum(np.exp(shifted), axis=0)) + beta * max_per_col  # (6,)
        smooth_max = log_sum / beta  # (6,)
        # violations[k] = smooth_max[k] - c_dot_n[k] - R ... wait, smooth_max already includes -c_dot_n
        # Actually diffs = v.n_k - c.n_k, so smooth_max of diffs = smooth_max(v.n_k) - c.n_k
        # violation = smooth_max - R
        violations = np.maximum(smooth_max - R, 0.0)  # (6,)
        # Objective: sum of exp(beta * violations) (soft) + small R penalty
        max_v = np.max(violations)
        if max_v > 30.0:
            return max_v * 100.0 + 1e10
        log_sum_v = np.log(np.sum(np.exp(beta * violations))) + beta * max_v
        return log_sum_v / beta + R * 0.001

    # Initial: center at mean, R at max support
    cx0 = np.mean(verts[:, 0])
    cy0 = np.mean(verts[:, 1])
    c_dot_n0 = cx0 * normals[:, 0] + cy0 * normals[:, 1]
    max_per_col0 = np.max(v_dot_n, axis=0)
    R0 = np.max(max_per_col0 - c_dot_n0) * 1.1 + 0.1

    from scipy.optimize import minimize
    try:
        res = minimize(objective, [cx0, cy0, R0], method='L-BFGS-B',
                       bounds=[(None, None), (None, None), (0, None)],
                       options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
        if res.success:
            cx, cy, R = res.x
            if R < 0:
                R = 0.0
            side = R / (_SQRT3 / 2)
            return cx, cy, side
    except Exception:
        pass
    return None

def _search_outer_angle_lp(verts, coarse_step=0.5, fine_step=0.1, finer_step=0.02, finest_step=0.005):
    """Multi-resolution angle sweep using exact LP. Returns (cx, cy, side, angle_deg)."""
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0

    for angle_deg in np.arange(0, 60, coarse_step):
        if time.time() > _DEADLINE - 10.0:
            break
        res = _min_enclosing_hex_side_lp(verts, float(angle_deg))
        if res:
            cx, cy, side = res
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)

    if best_side == float('inf'):
        return 0.0, 0.0, float('inf'), 0.0

    for angle_deg in np.arange(max(0.0, best_angle - 2.0), min(60.0, best_angle + 2.0), fine_step):
        if time.time() > _DEADLINE - 10.0:
            break
        res = _min_enclosing_hex_side_lp(verts, float(angle_deg))
        if res:
            cx, cy, side = res
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)

    for angle_deg in np.arange(max(0.0, best_angle - 0.5), min(60.0, best_angle + 0.5), finer_step):
        if time.time() > _DEADLINE - 10.0:
            break
        res = _min_enclosing_hex_side_lp(verts, float(angle_deg))
        if res:
            cx, cy, side = res
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)

    for angle_deg in np.arange(max(0.0, best_angle - 0.15), min(60.0, best_angle + 0.15), finest_step):
        if time.time() > _DEADLINE - 10.0:
            break
        res = _min_enclosing_hex_side_lp(verts, float(angle_deg))
        if res:
            cx, cy, side = res
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)

    return best_center[0], best_center[1], best_side, best_angle

def _polish_angle_lp(verts, ang0):
    """Golden-section + Nelder-Mead polish on angle using exact LP."""
    if time.time() > _DEADLINE - 15.0:
        return None

    def objective(ang_p):
        ang_p = float(ang_p % 60.0)
        res = _min_enclosing_hex_side_lp(verts, ang_p)
        if res is None:
            return 1e10
        return res[2]

    # Golden section
    try:
        a, b = max(0.0, ang0 - 0.05), min(60.0, ang0 + 0.05)
        gr = (math.sqrt(5.0) - 1.0) / 2.0
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = objective(c)
        fd = objective(d)
        for _ in range(100):
            if time.time() > _DEADLINE - 10.0:
                break
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = objective(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = objective(d)
        ang_best = float((a + b) / 2.0 % 60.0)
        res2 = _min_enclosing_hex_side_lp(verts, ang_best)
        if res2 is not None:
            return res2[0], res2[1], res2[2], ang_best
    except Exception:
        pass
    return None

def _polish_smooth(verts, ang0, cx0, cy0, R0_side, beta=80.0):
    """Full smooth polish: optimize (cx, cy, R) for a given angle, then refine angle.
    Returns (cx, cy, side, angle) or None."""
    if time.time() > _DEADLINE - 12.0:
        return None

    base_angle = math.radians(ang0 + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    v_dot_n = verts @ normals.T

    def objective(params):
        cx, cy, R = params
        if R < 0:
            R = 0.0
        c_dot_n = cx * normals[:, 0] + cy * normals[:, 1]
        diffs = v_dot_n - c_dot_n[np.newaxis, :]
        max_per_col = np.max(diffs, axis=0)
        shifted = beta * (diffs - max_per_col[np.newaxis, :])
        shifted = np.clip(shifted, -50.0, 50.0)
        log_sum = np.log(np.sum(np.exp(shifted), axis=0)) + beta * max_per_col
        smooth_max = log_sum / beta
        violations = np.maximum(smooth_max - R, 0.0)
        max_v = np.max(violations)
        if max_v > 30.0:
            return max_v * 100.0 + 1e10
        log_sum_v = np.log(np.sum(np.exp(beta * violations))) + beta * max_v
        return log_sum_v / beta + R * 0.001

    from scipy.optimize import minimize
    try:
        res = minimize(objective, [cx0, cy0, R0_side * _SQRT3 / 2], method='L-BFGS-B',
                       bounds=[(None, None), (None, None), (0, None)],
                       options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13})
        if res.success:
            cx, cy, R = res.x
            if R < 0:
                R = 0.0
            side = R / (_SQRT3 / 2)
            return cx, cy, side, ang0
    except Exception:
        pass
    return None

def _build_layouts():
    """Generate candidate layouts for 11 unit hexagons on triangular lattice.
    Returns list of (positions_list, inner_angle)."""
    layouts = []
    scale = 1.000001
    u = (_SQRT3 * scale, 0.0)
    v = (_SQRT3 / 2 * scale, 1.5 * scale)

    ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    ring2 = [(2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0),
             (-1, -1), (0, -2), (1, -2), (2, -1)]

    # Layout A: center + ring1 + best 4 from ring2 (by bbox)
    best_combo_side = float('inf')
    best_combo = None
    for combo in itertools.combinations(ring2, 4):
        target_test = [(0, 0)] + ring1 + list(combo)
        positions_test = []
        for ti, tj in target_test:
            x = ti * u[0] + tj * v[0]
            y = ti * u[1] + tj * v[1]
            positions_test.append((x, y))
        xs = [p[0] for p in positions_test]
        ys = [p[1] for p in positions_test]
        bbox = (max(xs) - min(xs)) + (max(ys) - min(ys))
        if bbox < best_combo_side:
            best_combo_side = bbox
            best_combo = combo

    if best_combo:
        target = [(0, 0)] + ring1 + list(best_combo)
        positions = []
        for ti, tj in target:
            x = ti * u[0] + tj * v[0]
            y = ti * u[1] + tj * v[1]
            positions.append((x, y))
        for inner_angle in [0.0, 30.0]:
            layouts.append((positions[:N], inner_angle))

    # Layout B: greedy nearest-neighbor on triangular lattice
    coords = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            coords.append((x, y))
    coords_sorted = sorted(coords, key=lambda t: t[0] ** 2 + t[1] ** 2)
    positions = coords_sorted[:N]
    for inner_angle in [0.0, 30.0]:
        layouts.append((positions, inner_angle))

    # Layout C: 3-4-4 rows
    positions = []
    y1 = -1.5 * scale
    for k in range(-1, 2):
        x = k * _SQRT3 * scale + _SQRT3 / 2 * scale
        positions.append((x, y1))
    y2 = 0.0
    for k in range(-1, 3):
        x = k * _SQRT3 * scale
        positions.append((x, y2))
    y3 = 1.5 * scale
    for k in range(-1, 3):
        x = k * _SQRT3 * scale + _SQRT3 / 2 * scale
        positions.append((x, y3))
    for inner_angle in [0.0, 30.0]:
        layouts.append((positions[:N], inner_angle))

    # Layout D: 4-3-4 rows
    positions = []
    y1 = -1.5 * scale
    for k in range(-2, 2):
        x = k * _SQRT3 * scale
        positions.append((x, y1))
    y2 = 0.0
    for k in range(-1, 2):
        x = k * _SQRT3 * scale + _SQRT3 / 2 * scale
        positions.append((x, y2))
    y3 = 1.5 * scale
    for k in range(-2, 2):
        x = k * _SQRT3 * scale
        positions.append((x, y3))
    for inner_angle in [0.0, 30.0]:
        layouts.append((positions[:N], inner_angle))

    return layouts

def construct():
    global _START_TIME, _DEADLINE

    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    layouts = _build_layouts()

    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0

    for positions, inner_angle in layouts:
        if time.time() > _DEADLINE - 12.0:
            break

        inner_hexes = [[pos[0], pos[1], inner_angle] for pos in positions]
        verts = _compute_all_vertices(inner_hexes)

        # Coarse + fine LP angle search
        cx, cy, side, angle = _search_outer_angle_lp(verts)

        if side < best_side:
            best_side = side
            best_inner = inner_hexes
            best_center = (cx, cy)
            best_outer_angle = angle

    if best_inner is None:
        scale = 1.000001
        u = (_SQRT3 * scale, 0.0)
        v = (_SQRT3 / 2 * scale, 1.5 * scale)
        coords = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = i * u[0] + j * v[0]
                y = i * u[1] + j * v[1]
                coords.append((x, y))
        coords_sorted = sorted(coords, key=lambda t: t[0] ** 2 + t[1] ** 2)
        best_inner = [[coords_sorted[i][0], coords_sorted[i][1], 0.0] for i in range(N)]
        verts = _compute_all_vertices(best_inner)
        cx, cy, side, angle = _search_outer_angle_lp(verts)
        best_side = side
        best_center = (cx, cy)
        best_outer_angle = angle

    # Polish: LP golden-section on angle
    if time.time() < _DEADLINE - 15.0:
        verts = _compute_all_vertices(best_inner)
        polished = _polish_angle_lp(verts, best_outer_angle)
        if polished is not None:
            cx_p, cy_p, side_p, ang_p = polished
            if side_p < best_side:
                best_side = side_p
                best_center = (cx_p, cy_p)
                best_outer_angle = ang_p

    # Smooth polish: optimize (cx, cy, R) jointly
    if time.time() < _DEADLINE - 12.0:
        verts = _compute_all_vertices(best_inner)
        R0 = best_side * _SQRT3 / 2
        smooth_res = _polish_smooth(verts, best_outer_angle, best_center[0], best_center[1], R0)
        if smooth_res is not None:
            cx_s, cy_s, side_s, ang_s = smooth_res
            if side_s < best_side:
                best_side = side_s
                best_center = (cx_s, cy_s)
                best_outer_angle = ang_s

    # Local search on inner positions to tighten packing
    if time.time() < _DEADLINE - 20.0:
        current_inner = [list(h) for h in best_inner]
        current_side = best_side
        current_center = best_center
        current_angle = best_outer_angle
        
        np.random.seed(42) # For reproducibility in search, though not strictly required
        
        for _ in range(1000):
            if time.time() > _DEADLINE - 10.0:
                break
            i = np.random.randint(N)
            dx = np.random.uniform(-0.05, 0.05)
            dy = np.random.uniform(-0.05, 0.05)
            
            trial_inner = [list(h) for h in current_inner]
            trial_inner[i][0] += dx
            trial_inner[i][1] += dy
            
            if _all_disjoint(trial_inner):
                verts = _compute_all_vertices(trial_inner)
                res = _min_enclosing_hex_side_lp(verts, current_angle)
                if res:
                    cx, cy, side = res
                    if side < current_side:
                        current_inner = trial_inner
                        current_side = side
                        current_center = (cx, cy)
                        
                        # Quick angle refinement
                        for ang_offset in [-0.5, 0.0, 0.5]:
                            ang_test = float((current_angle + ang_offset) % 60.0)
                            res_ang = _min_enclosing_hex_side_lp(verts, ang_test)
                            if res_ang and res_ang[2] < current_side:
                                current_side = res_ang[2]
                                current_center = (res_ang[0], res_ang[1])
                                current_angle = ang_test

        if current_side < best_side:
            best_inner = current_inner
            best_side = current_side
            best_center = current_center
            best_outer_angle = current_angle

    # Final LP verification and micro-polish
    if time.time() < _DEADLINE - 8.0:
        verts = _compute_all_vertices(best_inner)
        final = _min_enclosing_hex_side_lp(verts, best_outer_angle)
        if final is not None:
            cx_f, cy_f, side_f = final
            if side_f < best_side:
                best_side = side_f
                best_center = (cx_f, cy_f)

        # Very fine angle search around best
        for offset in np.linspace(-0.02, 0.02, 41):
            if time.time() > _DEADLINE - 8.0:
                break
            ang_test = float((best_outer_angle + offset) % 60.0)
            res = _min_enclosing_hex_side_lp(verts, ang_test)
            if res is not None:
                cx_t, cy_t, side_t = res
                if side_t < best_side:
                    best_side = side_t
                    best_center = (cx_t, cy_t)
                    best_outer_angle = ang_test

    return best_inner, best_center, best_side, best_outer_angle
