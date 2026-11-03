"""Direct hexagon packing: 11 unit hexagons with LP-optimized outer hexagon
and multi-start Nelder-Mead polish. Combines the structural lattice search
with LP-based exact enclosing-hexagon optimization."""
import math
import time
import itertools
import numpy as np
from scipy.optimize import minimize, linprog

N = 11


def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with side length `side`,
    centered at (cx, cy), rotated by angle_deg degrees."""
    verts = []
    for k in range(6):
        a = math.radians(angle_deg + 60 * k)
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return verts


def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with side `side` rotated by angle_deg."""
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


def _min_outer_side(hexes, ox, oy, oangle=0.0):
    """Find minimum side length of outer hexagon centered at (ox, oy) with rotation oangle
    that contains all hexagons in `hexes`."""
    max_val = 0.0
    a = -math.radians(oangle)
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    for cx, cy, s, ang in hexes:
        verts = _hex_vertices(cx, cy, s, ang)
        for vx, vy in verts:
            dx = vx - ox
            dy = vy - oy
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            for k in range(6):
                theta = math.radians(30 + 60 * k)
                proj = lx * math.cos(theta) + ly * math.sin(theta)
                if proj > max_val:
                    max_val = proj
    return max_val * 2.0 / math.sqrt(3)


def _lp_min_enclosing_hex(vertices, oangle=0.0):
    """Use LP to find the minimum side length and center of a regular hexagon
    with fixed rotation oangle that encloses all given vertices.
    
    Variables: cx, cy, s
    Minimize s subject to:
      For each vertex v and each of 6 normals n_k:
        n_k . (v - c) <= s * sqrt(3)/2
    """
    sqrt3 = math.sqrt(3)
    apothem_factor = sqrt3 / 2.0

    normals = []
    for k in range(6):
        theta = math.radians(30 + 60 * k)
        normals.append((math.cos(theta), math.sin(theta)))

    a = -math.radians(oangle)
    cos_a = math.cos(a)
    sin_a = math.sin(a)

    rotated_normals = []
    for nx, ny in normals:
        rx = nx * cos_a - ny * sin_a
        ry = nx * sin_a + ny * cos_a
        rotated_normals.append((rx, ry))

    n_vertices = len(vertices)
    n_constraints = n_vertices * 6

    c_obj = np.array([0.0, 0.0, 1.0])

    A_ub = np.zeros((n_constraints, 3))
    b_ub = np.zeros(n_constraints)

    idx = 0
    for vx, vy in vertices:
        for nx, ny in rotated_normals:
            A_ub[idx, 0] = -nx
            A_ub[idx, 1] = -ny
            A_ub[idx, 2] = -apothem_factor
            b_ub[idx] = -(nx * vx + ny * vy)
            idx += 1

    bounds = [(None, None), (None, None), (0.0, None)]

    try:
        result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if result.status == 0:
            return result.x[0], result.x[1], result.x[2]
    except Exception:
        pass

    # Fallback: centroid-based
    cx = sum(v[0] for v in vertices) / len(vertices)
    cy = sum(v[1] for v in vertices) / len(vertices)
    side = _min_outer_side([(0, 0, 1, 0)] * 0, cx, cy, oangle)
    # Recompute properly
    max_val = 0.0
    a2 = -math.radians(oangle)
    cos_a2 = math.cos(a2)
    sin_a2 = math.sin(a2)
    for vx, vy in vertices:
        dx = vx - cx
        dy = vy - cy
        lx = dx * cos_a2 - dy * sin_a2
        ly = dx * sin_a2 + dy * cos_a2
        for k in range(6):
            theta = math.radians(30 + 60 * k)
            proj = lx * math.cos(theta) + ly * math.sin(theta)
            if proj > max_val:
                max_val = proj
    side = max_val * 2.0 / sqrt3
    return cx, cy, side


def _optimize_outer_lp(inner_hexes, angles_to_try):
    """For a fixed set of inner hexagons, find the best outer hexagon
    by trying multiple angles and using LP for each."""
    all_verts = []
    for cx, cy, s, ang in inner_hexes:
        verts = _hex_vertices(cx, cy, s, ang)
        all_verts.extend(verts)

    best_side = float('inf')
    best_cx, best_cy, best_angle = 0.0, 0.0, 0.0

    for oang in angles_to_try:
        cx, cy, s = _lp_min_enclosing_hex(all_verts, oang)
        if s < best_side:
            best_side = s
            best_cx, best_cy, best_angle = cx, cy, oang

    return best_cx, best_cy, best_angle, best_side


def _nelder_mead(f, x0, max_iter=1000, tol=1e-12):
    """Nelder-Mead optimization. Returns (x_best, f_best)."""
    n = len(x0)
    simplex = [list(x0)]
    for i in range(n):
        x = list(x0)
        x[i] += 0.1
        simplex.append(x)

    fvals = [f(s) for s in simplex]

    for it in range(max_iter):
        order = sorted(range(n + 1), key=lambda i: fvals[i])
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]

        if abs(fvals[-1] - fvals[0]) < tol:
            break

        centroid = [sum(simplex[i][j] for i in range(n)) / n for j in range(n)]

        alpha = 1.0
        xr = [centroid[j] + alpha * (centroid[j] - simplex[-1][j]) for j in range(n)]
        fr = f(xr)

        if fvals[0] <= fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        elif fr < fvals[0]:
            gamma = 2.0
            xe = [centroid[j] + gamma * (centroid[j] - simplex[-1][j]) for j in range(n)]
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        else:
            rho = 0.5
            if fr < fvals[-1]:
                xc = [centroid[j] + rho * (simplex[-1][j] - centroid[j]) for j in range(n)]
            else:
                xc = [centroid[j] - rho * (centroid[j] - simplex[-1][j]) for j in range(n)]
            fc = f(xc)
            if fc < min(fr, fvals[-1]):
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = [(simplex[0][j] + simplex[i][j]) / 2 for j in range(n)]
                    fvals[i] = f(simplex[i])

    best_idx = min(range(n + 1), key=lambda i: fvals[i])
    return simplex[best_idx], fvals[best_idx]


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s limit

    sqrt3 = math.sqrt(3)

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
    best_center = (0.0, 0.0)
    best_angle = 0.0

    all_combos = list(itertools.combinations(range(len(ring2)), 4))

    def combo_score(combo):
        idxs = sorted(combo)
        gaps = []
        for i in range(len(idxs)):
            next_i = (i + 1) % len(idxs)
            gap = (idxs[next_i] - idxs[i]) % len(ring2)
            gaps.append(gap)
        return sum(gaps)

    all_combos.sort(key=combo_score)

    # Phase 1: Scan combos with LP-based outer optimization
    # Use a coarser angle grid first, then refine
    coarse_angles = list(np.arange(0.0, 180.0, 5.0))

    for combo in all_combos:
        if time.time() > deadline - 50.0:
            break

        selected_r2 = [ring2[i] for i in combo]
        full_axial = pos_axial + selected_r2
        full_xy = [axial_to_xy(q, r) for q, r in full_axial]
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in full_xy]

        cx, cy, ang, side = _optimize_outer_lp(inner_hexes, coarse_angles)
        if side < best_side:
            best_side = side
            best_positions = full_xy
            best_center = (cx, cy)
            best_angle = ang

    # Phase 2: Refine the best combo with finer angle grid + Nelder-Mead
    if best_positions is not None and time.time() < deadline:
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_positions]

        # Finer angle scan around the best angle
        fine_angles = list(np.arange(best_angle - 15.0, best_angle + 15.0, 1.0))
        cx, cy, ang, side = _optimize_outer_lp(inner_hexes, fine_angles)
        if side < best_side:
            best_side = side
            best_center = (cx, cy)
            best_angle = ang

        # Nelder-Mead on (cx, cy, angle) using the analytic _min_outer_side
        def objective(params):
            cx, cy, ang = params
            return _min_outer_side(inner_hexes, cx, cy, ang)

        x0 = [best_center[0], best_center[1], best_angle]
        x_best, f_best = _nelder_mead(objective, x0, max_iter=800, tol=1e-12)
        if f_best < best_side:
            best_side = f_best
            best_center = (x_best[0], x_best[1])
            best_angle = x_best[2]

        # Multi-start Nelder-Mead to escape local minima
        if time.time() < deadline - 15.0:
            cx_mean = sum(x for x, y in best_positions) / 11.0
            cy_mean = sum(y for x, y in best_positions) / 11.0
            for start_ang in [0.0, 30.0, -30.0, 15.0, -15.0, 45.0, -45.0, 60.0, -60.0]:
                if time.time() > deadline - 10.0:
                    break
                x0 = [cx_mean, cy_mean, start_ang]
                x_best, f_best = _nelder_mead(objective, x0, max_iter=400, tol=1e-12)
                if f_best < best_side:
                    best_side = f_best
                    best_center = (x_best[0], x_best[1])
                    best_angle = x_best[2]

    # Phase 3: Try a few other top combos with full optimization
    if best_positions is not None and time.time() < deadline - 20.0:
        # Re-sort and try next few combos that weren't the best
        tried = set()
        # We'll just try the next 5 best combos by score
        for combo in all_combos[:15]:
            if time.time() > deadline - 10.0:
                break
            # Skip if this is essentially the same as best (check by comparing)
            selected_r2 = [ring2[i] for i in combo]
            full_axial = pos_axial + selected_r2
            full_xy = [axial_to_xy(q, r) for q, r in full_axial]
            # Quick check: skip if same as current best
            if len(full_xy) == len(best_positions):
                same = True
                for p1, p2 in zip(full_xy, best_positions):
                    if abs(p1[0] - p2[0]) > 1e-6 or abs(p1[1] - p2[1]) > 1e-6:
                        same = False
                        break
                if same:
                    continue

            inner_hexes = [(x, y, 1.0, 0.0) for x, y in full_xy]
            cx, cy, ang, side = _optimize_outer_lp(inner_hexes, coarse_angles)
            if side < best_side:
                # Refine
                fine_angles = list(np.arange(ang - 10.0, ang + 10.0, 1.0))
                cx, cy, ang, side = _optimize_outer_lp(inner_hexes, fine_angles)
                if side < best_side:
                    def objective2(params):
                        cx2, cy2, ang2 = params
                        return _min_outer_side(inner_hexes, cx2, cy2, ang2)
                    x_best, f_best = _nelder_mead(objective2, [cx, cy, ang], max_iter=500, tol=1e-12)
                    if f_best < best_side:
                        best_side = f_best
                        best_positions = full_xy
                        best_center = (x_best[0], x_best[1])
                        best_angle = x_best[2]

    # Phase 4: Final high-precision polish
    if best_positions is not None and time.time() < deadline:
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_positions]

        def objective_final(params):
            cx, cy, ang = params
            return _min_outer_side(inner_hexes, cx, cy, ang)

        # LP-based final check
        all_verts = []
        for cx, cy, s, ang in inner_hexes:
            verts = _hex_vertices(cx, cy, s, ang)
            all_verts.extend(verts)

        # Try very fine angles around current best
        very_fine_angles = list(np.arange(best_angle - 5.0, best_angle + 5.0, 0.5))
        cx, cy, ang, side = _optimize_outer_lp(inner_hexes, very_fine_angles)
        if side < best_side:
            best_side = side
            best_center = (cx, cy)
            best_angle = ang

        # Nelder-Mead with very tight tolerance
        x0 = [best_center[0], best_center[1], best_angle]
        x_best, f_best = _nelder_mead(objective_final, x0, max_iter=1500, tol=1e-14)
        if f_best < best_side:
            best_side = f_best
            best_center = (x_best[0], x_best[1])
            best_angle = x_best[2]

        # SciPy L-BFGS-B as additional polish
        if time.time() < deadline - 5.0:
            try:
                res = minimize(objective_final, x0, method='L-BFGS-B',
                               bounds=[(-5, 5), (-5, 5), (-90, 90)],
                               options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-15})
                if res.success and res.fun < best_side:
                    best_side = res.fun
                    best_center = (res.x[0], res.x[1])
                    best_angle = res.x[2]
            except Exception:
                pass

    if best_positions is None:
        best_positions = [axial_to_xy(q, r) for q, r in pos_axial]
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_positions]
        cx, cy, ang, best_side = _optimize_outer_lp(inner_hexes, coarse_angles)
        best_center = (cx, cy)
        best_angle = ang

    # Build final inner hexagons
    inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_positions]

    # Verify disjointness
    if not _all_disjoint(inner_hexes):
        for scale in [1.001, 1.01, 1.05]:
            scaled_hexes = [(x * scale, y * scale, 1.0, 0.0) for x, y in best_positions]
            if _all_disjoint(scaled_hexes):
                inner_hexes = scaled_hexes
                best_positions = [(x, y) for x, y, s, a in inner_hexes]
                cx_mean = sum(x for x, y, s, a in inner_hexes) / 11.0
                cy_mean = sum(y for x, y, s, a in inner_hexes) / 11.0
                best_side = _min_outer_side(inner_hexes, cx_mean, cy_mean, best_angle)
                best_center = (cx_mean, cy_mean)
                break

    # Verify containment and binary-search for tightest valid side if needed
    outer_hex = (best_center[0], best_center[1], best_side, best_angle)
    valid = True
    for h in inner_hexes:
        if not _inner_in_outer(h, outer_hex):
            valid = False
            break

    if not valid:
        lo, hi = best_side, best_side * 1.1
        for _ in range(60):
            mid = (lo + hi) / 2
            test_outer = (best_center[0], best_center[1], mid, best_angle)
            ok = all(_inner_in_outer(h, test_outer) for h in inner_hexes)
            if ok:
                hi = mid
            else:
                lo = mid
        best_side = hi

    # Format output
    inner = [[h[0], h[1], h[3]] for h in inner_hexes]

    return inner, (best_center[0], best_center[1]), best_side, best_angle
