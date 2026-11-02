"""Combined program: LP-based outer hexagon + multi-layout search + polish."""
import math
import time
import numpy as np
import itertools

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0
_SQRT3 = math.sqrt(3)

def _hex_vertices_array(cx, cy, angle_deg):
    """Vertices of a unit regular hexagon centered at (cx,cy) with rotation angle_deg.
    Returns numpy array shape (6, 2)."""
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _compute_all_vertices(inner_hexes):
    """Precompute all vertices of inner hexagons as a numpy array (N*6, 2)."""
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
    """
    Find minimum side length of a regular hexagon (with vertex direction at outer_angle_deg)
    that contains all points in verts (numpy array of shape (M,2)).
    Returns (cx, cy, side) or None.
    Uses LP: minimize R s.t. for all vertices v and all 6 side normals n:
      (v - c) . n <= R  and  (c - v) . n <= R
    """
    M = len(verts)
    if M == 0:
        return 0.0, 0.0, 0.0

    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])

    # v_dot_n: (M, 6)
    v_dot_n = verts @ normals.T

    A_rows = []
    b_vals = []

    for k in range(6):
        nx, ny = normals[k]
        v_dots = v_dot_n[:, k]

        # (v - c).n <= R  =>  -nx*cx - ny*cy - R <= -v.n
        A_rows.append(np.column_stack([np.full(M, -nx), np.full(M, -ny), np.full(M, -1.0)]))
        b_vals.append(-v_dots)

        # (c - v).n <= R  =>  nx*cx + ny*cy - R <= v.n
        A_rows.append(np.column_stack([np.full(M, nx), np.full(M, ny), np.full(M, -1.0)]))
        b_vals.append(v_dots)

    A = np.vstack(A_rows)
    b = np.concatenate(b_vals)

    # R >= 0
    A = np.vstack([A, [[0.0, 0.0, -1.0]]])
    b = np.append(b, 0.0)

    c_obj = [0.0, 0.0, 1.0]
    try:
        from scipy.optimize import linprog
        res = linprog(c_obj, A_ub=A, b_ub=b,
                      bounds=[(None, None), (None, None), (0, None)],
                      method='highs')
        if res.success:
            cx, cy, R = res.x
            side = R / (_SQRT3 / 2)
            return cx, cy, side
    except Exception:
        pass
    return None

def _search_outer_angle(verts, coarse_step=2.0, fine_step=0.25, finer_step=0.05):
    """Multi-resolution angle sweep for the outer hexagon.
    Returns (cx, cy, side, angle_deg)."""
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # Coarse: 0 to 60 in steps of coarse_step
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

    # Fine
    for angle_deg in np.arange(max(0.0, best_angle - 5), min(60.0, best_angle + 5), fine_step):
        if time.time() > _DEADLINE - 10.0:
            break
        res = _min_enclosing_hex_side_lp(verts, float(angle_deg))
        if res:
            cx, cy, side = res
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)

    # Finer
    for angle_deg in np.arange(max(0.0, best_angle - 1.0), min(60.0, best_angle + 1.0), finer_step):
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

def _polish_with_nelder_mead(verts, cx0, cy0, ang0):
    """Polish (cx, cy, angle) using Nelder-Mead where each eval re-solves the LP.
    Returns improved (cx, cy, side, angle) or None."""
    if time.time() > _DEADLINE - 15.0:
        return None

    from scipy.optimize import minimize

    def objective(params):
        ang_p = params[2] % 60.0
        res = _min_enclosing_hex_side_lp(verts, float(ang_p))
        if res is None:
            return 1e10
        return res[2]

    x0 = np.array([cx0, cy0, ang0])
    try:
        res = minimize(objective, x0, method='Nelder-Mead',
                       options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-11,
                                'adaptive': True})
        if res.success:
            cx_nm, cy_nm, ang_nm = res.x
            ang_nm = float(ang_nm % 60.0)
            res2 = _min_enclosing_hex_side_lp(verts, ang_nm)
            if res2 is not None:
                return res2[0], res2[1], res2[2], ang_nm
    except Exception:
        pass
    return None

def _build_layouts():
    """Generate multiple candidate layouts for the 11 inner hexagons.
    Returns list of (positions_list, inner_angle)."""
    layouts = []
    scale = 1.000001
    u = (_SQRT3 * scale, 0.0)
    v = (_SQRT3 / 2 * scale, 1.5 * scale)

    # Layout 0: hexagonal flower + 4 from second ring (compact)
    target = [(0, 0)]
    ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    target.extend(ring1)
    ring2 = [(2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0),
             (-1, -1), (0, -2), (1, -2), (2, -1)]
    ring2_sorted = sorted(ring2, key=lambda p: p[0] ** 2 + p[1] ** 2)
    target.extend(ring2_sorted[:4])

    for inner_angle in [0.0, 30.0]:
        positions = []
        for ti, tj in target:
            x = ti * u[0] + tj * v[0]
            y = ti * u[1] + tj * v[1]
            positions.append((x, y))
        layouts.append((positions[:N], inner_angle))

    # Layout 1: row-based 3-4-4
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

    # Layout 2: 4-3-4 shifted
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

    # Layout 3: greedy nearest-neighbor from origin on triangular lattice
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

    # Layout 4: slightly different second-ring selection (pick 4 that minimize bounding)
    # Try different 4-of-10 combinations from ring2
    best_combo_side = float('inf')
    best_combo = None
    for combo in itertools.combinations(ring2, 4):
        target_test = [(0, 0)] + ring1 + list(combo)
        positions_test = []
        for ti, tj in target_test:
            x = ti * u[0] + tj * v[0]
            y = ti * u[1] + tj * v[1]
            positions_test.append((x, y))
        # Quick bounding check: use axis-aligned bounding box as proxy
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
        if time.time() > _DEADLINE - 10.0:
            break

        inner_hexes = [[pos[0], pos[1], inner_angle] for pos in positions]
        verts = _compute_all_vertices(inner_hexes)

        cx, cy, side, angle = _search_outer_angle(verts)

        if side < best_side:
            best_side = side
            best_inner = inner_hexes
            best_center = (cx, cy)
            best_outer_angle = angle

    # Polish the best layout
    if best_inner is not None and time.time() < _DEADLINE - 15.0:
        verts = _compute_all_vertices(best_inner)
        polished = _polish_with_nelder_mead(verts, best_center[0], best_center[1], best_outer_angle)
        if polished is not None:
            cx_p, cy_p, side_p, ang_p = polished
            if side_p < best_side:
                best_side = side_p
                best_center = (cx_p, cy_p)
                best_outer_angle = ang_p

    # Fallback
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
        cx, cy, side, angle = _search_outer_angle(verts)
        best_side = side
        best_center = (cx, cy)
        best_outer_angle = angle

    return best_inner, best_center, best_side, best_outer_angle
