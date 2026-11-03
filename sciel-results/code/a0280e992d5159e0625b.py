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

    A_rows = []
    b_vals = []

    for j in range(6):
        nx, ny = normals[j]
        # n . (v - C) <= h  =>  -n . C - h <= -n . v
        A_rows.append(np.tile([-nx, -ny, -1.0], (M, 1)))
        b_vals.append(-vdots[:, j])
        # -n . (v - C) <= h => n . C - h <= n . v
        # This is redundant if we have all 6 normals, but keep for safety/numerical issues?
        # Actually, for a regular hexagon, the 6 outward normals are sufficient.
        # The constraint n . (v-C) <= h ensures v is on the "inner" side of the face with normal n.
        # Since the hexagon is the intersection of these 6 half-planes, this is sufficient.
        # I will remove the second set of constraints to speed up the LP.
        
    # Add the redundant constraints back? No, they are redundant.
    # Let's stick to the 6 constraints only.
    
    A = np.vstack(A_rows)
    b = np.concatenate(b_vals)

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


def _smooth_objective_for_angle(verts, outer_angle_deg, beta=100.0):
    """
    Smooth surrogate for the outer side length using log-sum-exp.
    """
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])

    M = len(verts)
    vdots = verts @ normals.T  # shape (M, 6)

    # For a fixed angle, the min enclosing hex side is determined by
    # min_C max_k (max_i n_k . v_i - n_k . C)
    # This is equivalent to 0.5 * max_k (max_i n_k . v_i - min_i n_k . v_i)
    # IF the center C is free to move.
    # Wait, the width in direction n_k is max_i n_k . v_i - min_i n_k . v_i.
    # The minimum enclosing hexagon with fixed angle has side length
    # S = (2/sqrt(3)) * max_k (width_k / 2) ?
    # No. The hexagon is defined by 6 parallel pairs of lines.
    # The distance between the lines for normal n_k is width_k.
    # The apothem h must be at least width_k / 2 for all k.
    # So h_min = max_k (width_k / 2).
    # Side S = 2 * h_min / sqrt(3).
    
    widths = np.max(vdots, axis=0) - np.min(vdots, axis=0)  # shape (6,)
    R_vals = widths / 2.0

    max_R = np.max(R_vals)
    shifted = R_vals - max_R
    log_sum_exp = max_R + np.log(np.sum(np.exp(beta * shifted))) / beta

    side = 2.0 * log_sum_exp * _INVSQRT3
    return side


def _smooth_objective_grad(verts, outer_angle_deg, beta=100.0):
    """
    Smooth surrogate objective and its gradient with respect to outer_angle_deg.
    Returns (side, d_side_d_angle).
    """
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    d_normals = np.column_stack([-np.sin(angles), np.cos(angles)]) * (math.pi / 180.0)

    M = len(verts)
    vdots = verts @ normals.T
    dvdots = verts @ d_normals.T

    max_vals = np.max(vdots, axis=0)
    min_vals = np.min(vdots, axis=0)
    max_idx = np.argmax(vdots, axis=0)
    min_idx = np.argmin(vdots, axis=0)

    d_max = dvdots[np.arange(M), max_idx]
    d_min = dvdots[np.arange(M), min_idx]

    widths = max_vals - min_vals
    R_vals = widths / 2.0
    d_R_vals = (d_max - d_min) / 2.0

    max_R = np.max(R_vals)
    shifted = R_vals - max_R
    exp_shifted = np.exp(beta * shifted)
    sum_exp = np.sum(exp_shifted)

    log_sum_exp = max_R + np.log(sum_exp) / beta

    argmax_R = np.argmax(R_vals)
    d_max_R = d_R_vals[argmax_R]

    d_lse = d_max_R + np.sum(d_R_vals * exp_shifted) / (beta * sum_exp)

    side = 2.0 * log_sum_exp * _INVSQRT3
    d_side = 2.0 * d_lse * _INVSQRT3

    return side, d_side


def _build_positions(layout_idx):
    """Generate 11 positions for given layout index."""
    sqrt3 = _SQRT3
    scale = 1.001
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)

    if layout_idx == 0:
        # Compact: center + ring1 (6) + 4 from ring2
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
        # Fallback: nearest to origin
        coords = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = i * u[0] + j * v[0]
                y = i * u[1] + j * v[1]
                coords.append((x, y, i, j))
        coords_sorted = sorted(coords, key=lambda t: t[0] ** 2 + t[1] ** 2)
        return [(x, y) for x, y, i, j in coords_sorted[:N]]


def _optimize_outer_angle(verts, initial_angle, time_limit):
    """
    Optimize the outer hexagon angle using smooth surrogate + exact refinement.
    Returns (best_angle, best_side).
    """
    best_side = 1e10
    best_angle = initial_angle

    # Step 1: Smooth optimization via L-BFGS-B with analytic gradient
    try:
        def obj_and_grad(a):
            s, g = _smooth_objective_grad(verts, a[0], beta=100.0)
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

    # Step 3: Golden-section polish with exact objective
    if time.time() < time_limit:
        for delta_max, tol in [(0.01, 1e-10), (0.002, 1e-12)]:
            if time.time() > time_limit:
                break
            lo2 = max(0.0, best_angle - delta_max)
            hi2 = min(30.0, best_angle + delta_max)
            if hi2 <= lo2:
                continue
            # Golden section search
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


def construct():
    global _START_TIME, _DEADLINE

    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0

    num_layouts = 6
    layouts = [_build_positions(i) for i in range(num_layouts)]

    inner_angles = [0.0, 30.0, 15.0, -15.0, 7.5, -7.5]

    for layout_idx, positions in enumerate(layouts):
        if time.time() > _DEADLINE - 10.0:
            break

        for inner_angle in inner_angles:
            if time.time() > _DEADLINE - 8.0:
                break

            all_vertices = np.vstack(
                [_hex_vertices(px, py, inner_angle) for px, py in positions]
            )
            
            # Center the vertices to improve LP stability
            center_offset = np.mean(all_vertices, axis=0)
            centered_vertices = all_vertices - center_offset

            coarse_best = 1e10
            coarse_best_angle = 0.0
            coarse_results = []
            for outer_angle_deg in np.arange(0, 30.25, 1.0):
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
            for s, a in coarse_results[1:3]:
                if abs(a - candidate_starts[0]) > 0.5:
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
                    # Adjust center back
                    final_cx = cx + center_offset[0]
                    final_cy = cy + center_offset[1]
                    if side < best_side:
                        candidate_inner = [[pos[0], pos[1], inner_angle] for pos in positions]
                        candidate_center = (final_cx, final_cy)
                        if _validate_disjoint(candidate_inner, candidate_center, side, local_best_angle):
                            best_side = side
                            best_inner = candidate_inner
                            best_center = candidate_center
                            best_outer_angle = local_best_angle

    if best_inner is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    return best_inner, best_center, best_side, best_outer_angle


def _validate_disjoint(inner, center, side, outer_angle):
    """
    Validate that inner hexagons are strictly disjoint and contained in outer.
    Returns True if valid.
    """
    eps = 1e-9

    # Check inner hexagons are strictly disjoint using SAT
    for i in range(len(inner)):
        for j in range(i + 1, len(inner)):
            xi, yi, ai = inner[i]
            xj, yj, aj = inner[j]
            vi = _hex_vertices(xi, yi, ai)
            vj = _hex_vertices(xj, yj, aj)
            axes = []
            for poly in [vi, vj]:
                for k in range(6):
                    p1 = poly[k]
                    p2 = poly[(k + 1) % 6]
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm > 1e-15:
                        axes.append(normal / norm)
            separated = False
            for axis in axes:
                proj1 = vi @ axis
                proj2 = vj @ axis
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                if max1 < min2 - eps or max2 < min1 - eps:
                    separated = True
                    break
            if not separated:
                return False

    # Check all inner hexagons are inside outer
    outer_verts = _hex_vertices(center[0], center[1], outer_angle)
    outer_verts = center[0] + (outer_verts - center[0]) * side

    for i in range(len(inner)):
        xi, yi, ai = inner[i]
        vi = _hex_vertices(xi, yi, ai)
        for pt in vi:
            inside = True
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k + 1) % 6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                if np.dot(normal, pt - p1) > eps:
                    inside = False
                    break
            if not inside:
                return False

    return True
