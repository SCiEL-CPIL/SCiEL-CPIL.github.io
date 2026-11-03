"""Combined program: LP-based outer hexagon + joint rigid-body rotation + multi-layout search + polish."""
import math
import time
import numpy as np
import itertools

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0
_SQRT3 = math.sqrt(3)


def _hex_vertices_rotated(cx, cy, angle_deg):
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


def _rotate_points(verts, angle_deg, center=(0.0, 0.0)):
    """Rotate all points in verts (M,2) around center by angle_deg. Returns (M,2)."""
    if abs(angle_deg) < 1e-15:
        return verts
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    cx, cy = center
    # Translate to origin, rotate, translate back
    dx = verts[:, 0] - cx
    dy = verts[:, 1] - cy
    new_x = cx + c * dx - s * dy
    new_y = cy + s * dx + c * dy
    return np.column_stack([new_x, new_y])


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


def _eval_joint_rotation(base_verts, theta_deg, phi_deg):
    """
    Evaluate the enclosing hexagon side for a given inner-cluster rotation theta
    and outer hexagon angle phi. base_verts are the vertices at theta=0.
    Returns (cx, cy, side) or None.
    """
    rotated = _rotate_points(base_verts, theta_deg)
    return _min_enclosing_hex_side_lp(rotated, phi_deg)


def _search_joint_rotation(base_verts, theta_coarse=5.0, theta_fine=1.0, phi_coarse=1.0, phi_fine=0.2):
    """
    Multi-resolution sweep over joint (theta, phi) space.
    Returns (cx, cy, side, theta, phi).
    """
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_theta = 0.0
    best_phi = 0.0

    # Coarse sweep over theta (0 to 30 by symmetry) and phi (0 to 30)
    for theta in np.arange(0, 30.5, theta_coarse):
        if time.time() > _DEADLINE - 15.0:
            break
        for phi in np.arange(0, 30.5, phi_coarse):
            res = _eval_joint_rotation(base_verts, float(theta), float(phi))
            if res:
                cx, cy, side = res
                if side < best_side:
                    best_side = side
                    best_center = (cx, cy)
                    best_theta = float(theta)
                    best_phi = float(phi)

    if best_side == float('inf'):
        return 0.0, 0.0, float('inf'), 0.0, 0.0

    # Fine sweep around best
    for theta in np.arange(max(0.0, best_theta - 3), min(30.0, best_theta + 3), theta_fine):
        if time.time() > _DEADLINE - 15.0:
            break
        for phi in np.arange(max(0.0, best_phi - 3), min(30.0, best_phi + 3), phi_fine):
            res = _eval_joint_rotation(base_verts, float(theta), float(phi))
            if res:
                cx, cy, side = res
                if side < best_side:
                    best_side = side
                    best_center = (cx, cy)
                    best_theta = float(theta)
                    best_phi = float(phi)

    # Finer sweep
    for theta in np.arange(max(0.0, best_theta - 1.0), min(30.0, best_theta + 1.0), 0.2):
        if time.time() > _DEADLINE - 15.0:
            break
        for phi in np.arange(max(0.0, best_phi - 1.0), min(30.0, best_phi + 1.0), 0.2):
            res = _eval_joint_rotation(base_verts, float(theta), float(phi))
            if res:
                cx, cy, side = res
                if side < best_side:
                    best_side = side
                    best_center = (cx, cy)
                    best_theta = float(theta)
                    best_phi = float(phi)

    return best_center[0], best_center[1], best_side, best_theta, best_phi


def _polish_joint(base_verts, cx0, cy0, theta0, phi0):
    """Polish (theta, phi) using Nelder-Mead where each eval re-solves the LP.
    Returns improved (cx, cy, side, theta, phi) or None."""
    if time.time() > _DEADLINE - 20.0:
        return None

    from scipy.optimize import minimize

    def objective(params):
        theta_p = params[0] % 30.0
        phi_p = params[1] % 30.0
        res = _eval_joint_rotation(base_verts, float(theta_p), float(phi_p))
        if res is None:
            return 1e10
        return res[2]

    x0 = np.array([theta0, phi0])
    try:
        res = minimize(objective, x0, method='Nelder-Mead',
                       options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12,
                                'adaptive': True})
        if res.success:
            theta_nm, phi_nm = res.x
            theta_nm = float(theta_nm % 30.0)
            phi_nm = float(phi_nm % 30.0)
            res2 = _eval_joint_rotation(base_verts, theta_nm, phi_nm)
            if res2 is not None:
                return res2[0], res2[1], res2[2], theta_nm, phi_nm
    except Exception:
        pass
    return None


def _build_layouts():
    """Generate multiple candidate layouts for the 11 inner hexagons (at theta=0, angle=0).
    Returns list of positions lists (each is list of (x,y))."""
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

    positions = []
    for ti, tj in target:
        x = ti * u[0] + tj * v[0]
        y = ti * u[1] + tj * v[1]
        positions.append((x, y))
    layouts.append(positions[:N])

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
    layouts.append(positions[:N])

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
    layouts.append(positions[:N])

    # Layout 3: greedy nearest-neighbor from origin on triangular lattice
    coords = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            coords.append((x, y))
    coords_sorted = sorted(coords, key=lambda t: t[0] ** 2 + t[1] ** 2)
    positions = coords_sorted[:N]
    layouts.append(positions)

    # Layout 4: pick 4-of-10 from ring2 that minimize bounding box
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
        layouts.append(positions[:N])

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
    best_theta = 0.0
    best_phi = 0.0

    for positions in layouts:
        if time.time() > _DEADLINE - 20.0:
            break

        # Base vertices at theta=0, inner angle=0
        inner_hexes_base = [[pos[0], pos[1], 0.0] for pos in positions]
        base_verts = _compute_all_vertices(inner_hexes_base)

        cx, cy, side, theta, phi = _search_joint_rotation(base_verts)

        if side < best_side:
            best_side = side
            best_theta = theta
            best_phi = phi
            best_center = (cx, cy)
            # Build the actual inner hexagons with the optimized rotation
            # Each inner hexagon is rotated by theta around the cluster center
            # Cluster center is approximately the mean of positions
            px = sum(p[0] for p in positions) / N
            py = sum(p[1] for p in positions) / N
            inner_hexes = []
            for pos in positions:
                # Rotate position around cluster center by theta
                a = math.radians(theta)
                c, s = math.cos(a), math.sin(a)
                dx = pos[0] - px
                dy = pos[1] - py
                new_x = px + c * dx - s * dy
                new_y = py + s * dx + c * dy
                inner_hexes.append([new_x, new_y, theta])
            best_inner = inner_hexes

    # Polish the best layout
    if best_inner is not None and time.time() < _DEADLINE - 25.0:
        # Recompute base vertices for the best layout
        px = sum(p[0] for p in best_inner) / N
        py = sum(p[1] for p in best_inner) / N
        # Un-rotate to get base positions
        base_positions = []
        for pos in best_inner:
            a = math.radians(-best_theta)
            c, s = math.cos(a), math.sin(a)
            dx = pos[0] - px
            dy = pos[1] - py
            new_x = px + c * dx - s * dy
            new_y = py + s * dx + c * dy
            base_positions.append((new_x, new_y))
        inner_hexes_base = [[p[0], p[1], 0.0] for p in base_positions]
        base_verts = _compute_all_vertices(inner_hexes_base)

        polished = _polish_joint(base_verts, best_center[0], best_center[1], best_theta, best_phi)
        if polished is not None:
            cx_p, cy_p, side_p, theta_p, phi_p = polished
            if side_p < best_side:
                best_side = side_p
                best_center = (cx_p, cy_p)
                best_theta = theta_p
                best_phi = phi_p
                # Rebuild inner hexagons
                px = sum(p[0] for p in base_positions) / N
                py = sum(p[1] for p in base_positions) / N
                inner_hexes = []
                for pos in base_positions:
                    a = math.radians(theta_p)
                    c, s = math.cos(a), math.sin(a)
                    dx = pos[0] - px
                    dy = pos[1] - py
                    new_x = px + c * dx - s * dy
                    new_y = py + s * dx + c * dy
                    inner_hexes.append([new_x, new_y, theta_p])
                best_inner = inner_hexes

        # Second polish pass
        if time.time() < _DEADLINE - 15.0:
            polished2 = _polish_joint(base_verts, best_center[0], best_center[1], best_theta, best_phi)
            if polished2 is not None:
                cx_p2, cy_p2, side_p2, theta_p2, phi_p2 = polished2
                if side_p2 < best_side:
                    best_side = side_p2
                    best_center = (cx_p2, cy_p2)
                    best_theta = theta_p2
                    best_phi = phi_p2
                    px = sum(p[0] for p in base_positions) / N
                    py = sum(p[1] for p in base_positions) / N
                    inner_hexes = []
                    for pos in base_positions:
                        a = math.radians(theta_p2)
                        c, s = math.cos(a), math.sin(a)
                        dx = pos[0] - px
                        dy = pos[1] - py
                        new_x = px + c * dx - s * dy
                        new_y = py + s * dx + c * dy
                        inner_hexes.append([new_x, new_y, theta_p2])
                    best_inner = inner_hexes

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
        base_verts = _compute_all_vertices(best_inner)
        cx, cy, side, theta, phi = _search_joint_rotation(base_verts)
        best_side = side
        best_center = (cx, cy)
        best_theta = theta
        best_phi = phi

    return best_inner, best_center, best_side, best_phi

