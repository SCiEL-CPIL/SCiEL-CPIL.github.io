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
    Uses LP with support function.
    """
    M = len(verts)
    if M == 0:
        return 0.0, 0.0, 0.0

    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])  # (6, 2)

    v_dot_n = verts @ normals.T  # (M, 6)
    max_dots = np.max(v_dot_n, axis=0)  # (6,)
    min_dots = np.min(v_dot_n, axis=0)  # (6,)

    # LP: minimize R
    # s.t. for each normal k:
    #   -nx*cx - ny*cy - R <= -max_dots[k]
    #    nx*cx + ny*cy - R <=  min_dots[k]
    #   R >= 0

    A = np.empty((13, 3))
    b = np.empty(13)
    for k in range(6):
        nx, ny = normals[k]
        A[2*k, :] = [-nx, -ny, -1.0]
        b[2*k] = -max_dots[k]
        A[2*k+1, :] = [nx, ny, -1.0]
        b[2*k+1] = min_dots[k]
    A[12, :] = [0.0, 0.0, -1.0]
    b[12] = 0.0

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

def _min_enclosing_hex_side_smooth(verts, outer_angle_deg, beta=50.0):
    """
    Smooth approximation of the min enclosing hexagon side using log-sum-exp.
    Returns (cx, cy, side) or None.
    """
    M = len(verts)
    if M == 0:
        return 0.0, 0.0, 0.0

    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])

    v_dot_n = verts @ normals.T  # (M, 6)

    def objective(params):
        cx, cy, R = params
        if R < 0:
            R = 0.0
        # Constraints: for each vertex v and normal k: | (v-c).n_k | <= R
        # Soft constraint: sum exp(beta * max(0, |(v-c).n_k| - R))
        diffs = v_dot_n - (cx * normals[0, 0] + cy * normals[0, 1])  # wrong, need per-normal
        # Actually: (v - c) . n_k = v.n_k - (cx*n_kx + cy*n_ky)
        c_dot_n = np.array([cx * normals[k, 0] + cy * normals[k, 1] for k in range(6)])
        # diffs[m, k] = v_dot_n[m, k] - c_dot_n[k]
        diffs = v_dot_n - c_dot_n[np.newaxis, :]
        violations = np.abs(diffs) - R
        violations = np.maximum(violations, 0.0)
        # Log-sum-exp for numerical stability
        max_v = np.max(violations)
        if max_v > 50.0:
            return max_v * 10.0 + 1e10
        log_sum = np.log(np.sum(np.exp(beta * violations))) + beta * max_v
        return log_sum / beta + R * 0.01  # slight penalty on R

    # Initial guess: center at mean, R at max distance
    cx0 = np.mean(verts[:, 0])
    cy0 = np.mean(verts[:, 1])
    dists = np.sqrt((verts[:, 0] - cx0)**2 + (verts[:, 1] - cy0)**2)
    R0 = np.max(dists) * 1.2

    from scipy.optimize import minimize
    try:
        res = minimize(objective, [cx0, cy0, R0], method='L-BFGS-B',
                       bounds=[(None, None), (None, None), (0, None)],
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
        if res.success:
            cx, cy, R = res.x
            if R < 0:
                R = 0.0
            side = R / (_SQRT3 / 2)
            return cx, cy, side
    except Exception:
        pass
    return None

def _search_outer_angle(verts):
    """Optimize outer hexagon angle using direct optimization.
    Returns (cx, cy, side, angle_deg)."""
    from scipy.optimize import minimize, differential_evolution

    def objective(angle_arr):
        ang = float(angle_arr[0]) % 60.0
        res = _min_enclosing_hex_side_lp(verts, ang)
        if res is None:
            return 1e10
        return res[2]

    # Coarse global search with differential evolution
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0

    if time.time() < _DEADLINE - 30.0:
        try:
            de_res = differential_evolution(objective, bounds=[(0, 60)],
                                             seed=42, maxiter=50, tol=1e-10,
                                             popsize=15, polish=False)
            if de_res.fun < best_side:
                best_angle = float(de_res.x[0]) % 60.0
                r = _min_enclosing_hex_side_lp(verts, best_angle)
                if r:
                    best_side = r[2]
                    best_center = (r[0], r[1])
        except Exception:
            pass

    # If DE failed, do a coarse grid
    if best_side == float('inf'):
        for angle_deg in np.arange(0, 60, 1.0):
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

    # Nelder-Mead polish
    if time.time() < _DEADLINE - 15.0:
        try:
            nm_res = minimize(objective, np.array([best_angle]), method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-14, 'fatol': 1e-16,
                                       'adaptive': True})
            if nm_res.fun < best_side:
                best_angle = float(nm_res.x[0]) % 60.0
                r = _min_enclosing_hex_side_lp(verts, best_angle)
                if r:
                    best_side = r[2]
                    best_center = (r[0], r[1])
        except Exception:
            pass

    # Golden section for final precision
    if time.time() < _DEADLINE - 5.0:
        try:
            a, b = max(0.0, best_angle - 0.01), min(60.0, best_angle + 0.01)
            gr = (math.sqrt(5.0) - 1.0) / 2.0
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            fc = objective([c])
            fd = objective([d])
            for _ in range(100):
                if time.time() > _DEADLINE - 5.0:
                    break
                if fc < fd:
                    b = d
                    d = c
                    fd = fc
                    c = b - gr * (b - a)
                    fc = objective([c])
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + gr * (b - a)
                    fd = objective([d])
            ang_best = float((a + b) / 2.0 % 60.0)
            r = _min_enclosing_hex_side_lp(verts, ang_best)
            if r and r[2] < best_side:
                best_side = r[2]
                best_center = (r[0], r[1])
                best_angle = ang_best
        except Exception:
            pass

    return best_center[0], best_center[1], best_side, best_angle

def _polish_with_nelder_mead(verts, cx0, cy0, ang0):
    """Polish the outer angle with high-precision Nelder-Mead + golden section.
    Returns improved (cx, cy, side, angle) or None."""
    if time.time() > _DEADLINE - 15.0:
        return None

    from scipy.optimize import minimize

    def objective(params):
        ang_p = float(params[0]) % 60.0
        res = _min_enclosing_hex_side_lp(verts, ang_p)
        if res is None:
            return 1e10
        return res[2]

    try:
        res = minimize(objective, np.array([ang0]), method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-15, 'fatol': 1e-17,
                                'adaptive': True, 'initial_simplex': None})
        if res.success or res.fun < 1e9:
            ang_nm = float(res.x[0]) % 60.0
            res2 = _min_enclosing_hex_side_lp(verts, ang_nm)
            if res2 is not None:
                # Golden section refinement
                a, b = max(0.0, ang_nm - 0.005), min(60.0, ang_nm + 0.005)
                gr = (math.sqrt(5.0) - 1.0) / 2.0
                c = b - gr * (b - a)
                d = a + gr * (b - a)
                fc = objective([c])
                fd = objective([d])
                for _ in range(120):
                    if time.time() > _DEADLINE - 5.0:
                        break
                    if fc < fd:
                        b = d
                        d = c
                        fd = fc
                        c = b - gr * (b - a)
                        fc = objective([c])
                    else:
                        a = c
                        c = d
                        fc = fd
                        d = a + gr * (b - a)
                        fd = objective([d])
                ang_best = float((a + b) / 2.0 % 60.0)
                res3 = _min_enclosing_hex_side_lp(verts, ang_best)
                if res3 is not None:
                    return res3[0], res3[1], res3[2], ang_best
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

    ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    ring2 = [(2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0),
             (-1, -1), (0, -2), (1, -2), (2, -1)]

    # Try all 4-of-10 combinations from ring2 (210 combos)
    # Use a quick proxy: minimize the max distance from center
    best_combo = None
    best_proxy = float('inf')
    for combo in itertools.combinations(ring2, 4):
        target_test = [(0, 0)] + ring1 + list(combo)
        max_d2 = 0
        for ti, tj in target_test:
            x = ti * u[0] + tj * v[0]
            y = ti * u[1] + tj * v[1]
            d2 = x * x + y * y
            if d2 > max_d2:
                max_d2 = d2
        if max_d2 < best_proxy:
            best_proxy = max_d2
            best_combo = combo

    # Also try combos that minimize spread (bbox diagonal)
    best_combo2 = None
    best_proxy2 = float('inf')
    for combo in itertools.combinations(ring2, 4):
        target_test = [(0, 0)] + ring1 + list(combo)
        xs, ys = [], []
        for ti, tj in target_test:
            x = ti * u[0] + tj * v[0]
            y = ti * u[1] + tj * v[1]
            xs.append(x)
            ys.append(y)
        diag = (max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2
        if diag < best_proxy2:
            best_proxy2 = diag
            best_combo2 = combo

    # Add layouts with different ring2 selections and inner angles
    combos_to_try = set()
    if best_combo:
        combos_to_try.add(best_combo)
    if best_combo2 and best_combo2 != best_combo:
        combos_to_try.add(best_combo2)
    # Also add the original sorted selection
    ring2_sorted = sorted(ring2, key=lambda p: p[0] ** 2 + p[1] ** 2)
    combos_to_try.add(tuple(ring2_sorted[:4]))

    for combo in combos_to_try:
        target = [(0, 0)] + ring1 + list(combo)
        for inner_angle in [0.0, 30.0, 15.0, 45.0]:
            positions = []
            for ti, tj in target:
                x = ti * u[0] + tj * v[0]
                y = ti * u[1] + tj * v[1]
                positions.append((x, y))
            layouts.append((positions[:N], inner_angle))

    # Row-based layouts
    for inner_angle in [0.0, 30.0, 15.0, 45.0]:
        # 3-4-4
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
        layouts.append((positions[:N], inner_angle))

        # 4-3-4
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
        layouts.append((positions[:N], inner_angle))

    # Nearest-neighbor on lattice
    coords = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            coords.append((x, y))
    coords_sorted = sorted(coords, key=lambda t: t[0] ** 2 + t[1] ** 2)
    for inner_angle in [0.0, 30.0, 15.0, 45.0]:
        layouts.append((coords_sorted[:N], inner_angle))

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

    # Polish the best layout with multiple passes
    if best_inner is not None:
        verts = _compute_all_vertices(best_inner)
        for pass_num in range(4):
            if time.time() > _DEADLINE - 10.0:
                break
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
