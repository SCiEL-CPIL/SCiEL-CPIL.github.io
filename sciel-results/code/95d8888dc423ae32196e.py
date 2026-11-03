import math
import time
import numpy as np
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0
_SQRT3 = math.sqrt(3)
_INVSQRT3 = 1.0 / _SQRT3


def _hex_vertices(cx, cy, angle_deg):
    """Vertices of a unit regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])


def _min_enclosing_hex_side(verts, outer_angle_deg):
    """
    Find minimum side length of a regular hexagon (with vertex at outer_angle_deg)
    that contains all points in verts (numpy array of shape (M,2)).
    Returns (cx, cy, side) or None.
    """
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])

    M = len(verts)
    vdots = verts @ normals.T  # shape (M, 6)

    # A_ub shape: (2*M*6, 3)
    # b_ub shape: (2*M*6,)
    
    # We can construct A and b more efficiently
    # For each normal j:
    #   -nx*cx - ny*cy - R <= -v_dot
    #    nx*cx + ny*cy - R <=  v_dot
    
    A_rows = []
    b_vals = []
    
    for j in range(6):
        nx, ny = normals[j]
        v_dots = vdots[:, j]
        
        # Row 1: -nx*cx - ny*cy - R <= -v_dot
        A_rows.append(np.column_stack([np.full(M, -nx), np.full(M, -ny), np.full(M, -1.0)]))
        b_vals.append(-v_dots)
        
        # Row 2: nx*cx + ny*cy - R <= v_dot
        A_rows.append(np.column_stack([np.full(M, nx), np.full(M, ny), np.full(M, -1.0)]))
        b_vals.append(v_dots)
        
    A = np.vstack(A_rows)
    b = np.concatenate(b_vals)

    # R >= 0
    A = np.vstack([A, np.array([[0.0, 0.0, -1.0]])])
    b = np.concatenate([b, [0.0]])

    c_obj = np.array([0.0, 0.0, 1.0])
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b,
                      bounds=[(None, None), (None, None), (0, None)],
                      method='highs',
                      options={'presolve': True,
                               'dual_feasibility_tolerance': 1e-12,
                               'primal_feasibility_tolerance': 1e-12})
        if res.success:
            cx, cy, R = res.x
            side = R * _INVSQRT3 * 2.0
            return cx, cy, side
    except Exception:
        pass
    return None


def _smooth_objective_2d(verts, cx, cy, outer_angle_deg, beta=100.0):
    """
    Smooth surrogate for the outer side length with explicit center (cx, cy).
    Returns (objective_value, gradient_wrt_cx, gradient_wrt_cy, gradient_wrt_angle_deg).
    """
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])

    M = len(verts)
    centered = verts - np.array([cx, cy])
    vdots = centered @ normals.T  # shape (M, 6)

    widths = np.max(vdots, axis=0) - np.min(vdots, axis=0)  # shape (6,)
    R_vals = widths / 2.0

    max_R = np.max(R_vals)
    shifted = R_vals - max_R
    log_sum_exp = max_R + np.log(np.sum(np.exp(beta * shifted))) / beta

    side = 2.0 * log_sum_exp * _INVSQRT3

    # Gradients
    weights = np.exp(beta * shifted)
    weights /= weights.sum()

    max_idx = np.argmax(vdots, axis=0)
    min_idx = np.argmin(vdots, axis=0)

    n_max = normals[max_idx]  # shape (6, 2)
    n_min = normals[min_idx]  # shape (6, 2)
    dR_dc = 0.5 * (n_max - n_min)  # shape (6, 2)
    dside_dc = 2.0 * _INVSQRT3 * (weights @ dR_dc)  # shape (2,)
    g_cx = -dside_dc[0]
    g_cy = -dside_dc[1]

    # Gradient w.r.t. angle
    dn_dtheta = np.column_stack([-np.sin(angles), np.cos(angles)])
    v_max = verts[max_idx]
    v_min = verts[min_idx]
    dR_dtheta = 0.5 * ((v_max - v_min) @ dn_dtheta.T)
    dLSE_dtheta = np.sum(weights * dR_dtheta)
    dside_dtheta = 2.0 * _INVSQRT3 * dLSE_dtheta
    dside_ddeg = dside_dtheta * math.pi / 180.0

    return side, g_cx, g_cy, dside_ddeg


def _optimize_outer_angle(verts, initial_angle, time_limit):
    """
    Optimize the outer hexagon angle using smooth surrogate + exact refinement.
    Returns (best_angle, best_side).
    """
    best_side = 1e10
    best_angle = initial_angle

    # Step 1: Smooth optimization via L-BFGS-B
    try:
        def obj_and_grad(a):
            s, gx, gy, g = _smooth_objective_2d(verts, 0.0, 0.0, a[0], beta=100.0)
            return s, np.array([g])

        res_opt = minimize(
            obj_and_grad,
            x0=[initial_angle],
            method='L-BFGS-B',
            bounds=[(0.0, 30.0)],
            jac=True,
            options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14}
        )
        if res_opt.success:
            opt_angle = float(np.clip(res_opt.x[0], 0.0, 30.0))
            if time.time() < time_limit:
                exact_side = _min_enclosing_hex_side(verts, opt_angle)
                if exact_side is not None:
                    if exact_side[2] < best_side:
                        best_side = exact_side[2]
                        best_angle = opt_angle
    except Exception:
        pass

    # Step 2: Bounded scalar minimization with exact objective
    lo = max(0.0, best_angle - 5.0)
    hi = min(30.0, best_angle + 5.0)
    try:
        res_scalar = minimize_scalar(
            lambda a: _min_enclosing_hex_side(verts, a)[2] if _min_enclosing_hex_side(verts, a) else 1e6,
            bounds=(lo, hi),
            method='bounded',
            options={'xatol': 1e-12, 'maxiter': 300}
        )
        if res_scalar.success:
            if res_scalar.fun < best_side:
                best_side = res_scalar.fun
                best_angle = float(res_scalar.x)
    except Exception:
        pass

    # Step 3: Golden-section polish
    if time.time() < time_limit:
        for delta_max, tol in [(0.01, 1e-11), (0.001, 1e-13), (0.0001, 1e-14)]:
            if time.time() > time_limit:
                break
            lo2 = max(0.0, best_angle - delta_max)
            hi2 = min(30.0, best_angle + delta_max)
            if hi2 <= lo2:
                continue
            gr = (math.sqrt(5) - 1) / 2
            a = lo2
            b = hi2
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            fc = _min_enclosing_hex_side(verts, c)
            fd = _min_enclosing_hex_side(verts, d)
            fc = fc[2] if fc else 1e6
            fd = fd[2] if fd else 1e6
            while (b - a) > tol and time.time() < time_limit:
                if fc < fd:
                    b = d
                    d = c
                    fd = fc
                    c = b - gr * (b - a)
                    fc = _min_enclosing_hex_side(verts, c)
                    fc = fc[2] if fc else 1e6
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + gr * (b - a)
                    fd = _min_enclosing_hex_side(verts, d)
                    fd = fd[2] if fd else 1e6
            mid = (a + b) / 2
            res = _min_enclosing_hex_side(verts, mid)
            if res:
                s = res[2]
                if s < best_side:
                    best_side = s
                    best_angle = float(mid)

    return best_angle, best_side


def _polish_center_angle(verts, init_cx, init_cy, init_angle, time_limit):
    """
    Jointly optimize outer hexagon center (cx, cy) and angle.
    """
    try:
        def obj_and_grad(x):
            cx, cy, ang = x
            s, gcx, gcy, gdeg = _smooth_objective_2d(verts, cx, cy, ang, beta=200.0)
            return s, np.array([gcx, gcy, gdeg])

        res = minimize(
            obj_and_grad,
            x0=[init_cx, init_cy, init_angle],
            method='L-BFGS-B',
            bounds=[(-20.0, 20.0), (-20.0, 20.0), (0.0, 30.0)],
            jac=True,
            options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-16}
        )
        if res.success:
            cx, cy, ang = res.x
            ang = float(np.clip(ang, 0.0, 30.0))
            exact = _min_enclosing_hex_side(verts, ang)
            if exact is not None:
                return exact[0], exact[1], ang, exact[2]
    except Exception:
        pass
    return None


def _build_positions(layout_idx):
    """Generate 11 positions for given layout index."""
    sqrt3 = _SQRT3
    scale = 1.0000001
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)

    if layout_idx == 0:
        # Compact: center + ring1 (6) + 4 from ring2 (symmetric)
        target = [(0, 0)]
        ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        target.extend(ring1)
        # Ring 2 symmetric: (2,0), (-1,2), (-2,-1), (1,-2) ? No, let's pick evenly spaced
        # Indices in ring2: 0:(2,0), 1:(1,1), 2:(0,2), 3:(-1,2), 4:(-2,1), 5:(-2,0), 6:(-1,-1), 7:(0,-2), 8:(1,-2), 9:(2,-1) ... 
        # Actually, let's just use the 4 that are furthest apart in angle
        ring2 = [(2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0),
                 (-1, -1), (0, -2), (1, -2), (2, -1)]
        # This list has 10 items. Missing 2?
        # Axial coords for ring 2:
        # (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0)
        # (-1,-1), (0,-2), (1,-2), (2,-1) -- wait, (2,-1) is (2,-1).
        # Let's generate all 12 properly.
        ring2_full = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                if abs(i) == 2 and abs(j) == 2 and i+j == 0: # Corner? No.
                    pass
                # Hex distance 2
                if abs(i) + abs(j) + abs(-i-j) == 4:
                    ring2_full.append((i, j))
        # Sort by angle to pick evenly spaced
        ring2_full.sort(key=lambda p: math.atan2(p[1]*1.5, p[0]*sqrt3/2 + p[1]*sqrt3/2))
        # Pick 0, 3, 6, 9
        picks = [0, 3, 6, 9]
        for p in picks:
            target.append(ring2_full[p])
            
        positions = []
        for ti, tj in target:
            x = ti * u[0] + tj * v[0]
            y = ti * u[1] + tj * v[1]
            positions.append((x, y))
        return positions[:N]
    elif layout_idx == 1:
        # 3-4-4 rows
        positions = []
        y1 = -1.5 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 3):
            x = k * sqrt3 * scale
            positions.append((x, y2))
        y3 = 1.5 * scale
        for k in range(-1, 3):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 2:
        # 4-3-4 rows
        positions = []
        y1 = -1.5 * scale
        for k in range(-2, 2):
            x = k * sqrt3 * scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 2):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y2))
        y3 = 1.5 * scale
        for k in range(-2, 2):
            x = k * sqrt3 * scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 3:
        # Staggered 2-4-2-3
        positions = []
        y1 = -3.0 * scale
        for k in range(0, 2):
            x = k * sqrt3 * scale - sqrt3 / 2 * scale
            positions.append((x, y1))
        y2 = -1.5 * scale
        for k in range(-1, 3):
            x = k * sqrt3 * scale
            positions.append((x, y2))
        y3 = 0.0
        for k in range(-1, 1):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y3))
        y4 = 1.5 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale
            positions.append((x, y4))
        return positions[:N]
    elif layout_idx == 4:
        # 3-5-3 rows
        positions = []
        y1 = -1.5 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-2, 3):
            x = k * sqrt3 * scale
            positions.append((x, y2))
        y3 = 1.5 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y3))
        return positions[:N]
    else:
        # Fallback
        coords = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = i * u[0] + j * v[0]
                y = i * u[1] + j * v[1]
                coords.append((x, y, i, j))
        coords_sorted = sorted(coords, key=lambda t: t[0] ** 2 + t[1] ** 2)
        return [(x, y) for x, y, i, j in coords_sorted[:N]]


def construct():
    global _START_TIME, _DEADLINE

    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0

    num_layouts = 5
    layouts = [_build_positions(i) for i in range(num_layouts)]

    inner_angles = [0.0, 30.0, 15.0, -15.0]

    for layout_idx, positions in enumerate(layouts):
        if time.time() > _DEADLINE - 10.0:
            break

        for inner_angle in inner_angles:
            if time.time() > _DEADLINE - 8.0:
                break

            all_vertices = np.vstack(
                [_hex_vertices(px, py, inner_angle) for px, py in positions]
            )
            
            center_offset = np.mean(all_vertices, axis=0)
            centered_vertices = all_vertices - center_offset

            coarse_best = 1e10
            coarse_best_angle = 0.0
            coarse_results = []
            # Finer coarse grid: 0.25 degree steps
            for outer_angle_deg in np.arange(0, 30.25, 0.25):
                if time.time() > _DEADLINE - 8.0:
                    break
                res = _min_enclosing_hex_side(centered_vertices, outer_angle_deg)
                if res:
                    coarse_results.append((res[2], outer_angle_deg))
                    if res[2] < coarse_best:
                        coarse_best = res[2]
                        coarse_best_angle = outer_angle_deg

            if not coarse_results:
                continue

            coarse_results.sort(key=lambda x: x[0])
            candidate_starts = [coarse_results[0][1]]
            for s, a in coarse_results[1:8]:
                if abs(a - candidate_starts[0]) > 0.25:
                    candidate_starts.append(a)

            local_best_side = coarse_best
            local_best_angle = coarse_best_angle

            for start_a in candidate_starts:
                if time.time() > _DEADLINE - 12.0:
                    break
                ang, side = _optimize_outer_angle(centered_vertices, start_a, _DEADLINE - 15.0)
                if side < local_best_side:
                    local_best_side = side
                    local_best_angle = ang

            if local_best_side < best_side:
                res_final = _min_enclosing_hex_side(centered_vertices, local_best_angle)
                if res_final:
                    cx, cy, side = res_final
                    final_cx = cx + center_offset[0]
                    final_cy = cy + center_offset[1]
                    if side < best_side:
                        best_side = side
                        best_inner = [[pos[0], pos[1], inner_angle] for pos in positions]
                        best_center = (final_cx, final_cy)
                        best_outer_angle = local_best_angle

                        if time.time() < _DEADLINE - 20.0:
                            polish_res = _polish_center_angle(
                                all_vertices, final_cx, final_cy, local_best_angle,
                                _DEADLINE - 25.0
                            )
                            if polish_res is not None:
                                p_cx, p_cy, p_ang, p_side = polish_res
                                if p_side < best_side:
                                    best_side = p_side
                                    best_center = (p_cx, p_cy)
                                    best_outer_angle = p_ang

    if best_inner is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    return best_inner, best_center, best_side, best_outer_angle
