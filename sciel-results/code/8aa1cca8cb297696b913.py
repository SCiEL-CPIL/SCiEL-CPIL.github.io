"""
Packing 11 unit regular hexagons inside a regular hexagon.
Strategy: Compact hexagonal lattice (7 center + 4 ring-2), optimize outer
hexagon orientation via piecewise-linear analysis and local refinement.
"""
import math
import time
import itertools
import numpy as np
from scipy.optimize import linprog

N = 11
SQRT3 = math.sqrt(3)
TIME_LIMIT = 120.0


def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for k in range(6):
        a = math.radians(angle_deg + 60 * k)
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return verts


def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
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


def _hexagons_disjoint(hex1, hex2, eps=1e-9):
    """Check if two hexagons have disjoint interiors using SAT."""
    cx1, cy1, s1, a1 = hex1
    cx2, cy2, s2, a2 = hex2
    dx = cx2 - cx1
    dy = cy2 - cy1
    if dx * dx + dy * dy > (s1 + s2) ** 2 + 1e-12:
        return True
    verts1 = np.array(_hex_vertices(cx1, cy1, s1, a1))
    verts2 = np.array(_hex_vertices(cx2, cy2, s2, a2))
    for vx, vy in verts1:
        if _point_in_hex(vx, vy, cx2, cy2, s2, a2):
            return False
    for vx, vy in verts2:
        if _point_in_hex(vx, vy, cx1, cy1, s1, a1):
            return False
    for verts, other in [(verts1, verts2), (verts2, verts1)]:
        for i in range(6):
            ax, ay = verts[i]
            bx, by = verts[(i + 1) % 6]
            ex = bx - ax
            ey = by - ay
            length = math.sqrt(ex * ex + ey * ey)
            if length < 1e-15:
                continue
            nx = -ey / length
            ny = ex / length
            proj1 = verts[:, 0] * nx + verts[:, 1] * ny
            proj2 = other[:, 0] * nx + other[:, 1] * ny
            if proj1.max() <= proj2.min() + eps or proj2.max() <= proj1.min() + eps:
                return True
    return False


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


def _min_outer_side_analytical(hexes, oangle_rad):
    """
    Find minimum side length of outer hexagon with rotation oangle_rad
    that contains all hexagons in `hexes` (centered at origin).
    Uses linear programming. Returns (min_side, center_x, center_y).
    """
    points = []
    for cx, cy, s, ang in hexes:
        verts = _hex_vertices(cx, cy, s, ang)
        points.extend(verts)

    if not points:
        return 0.0, 0.0, 0.0

    pts = np.array(points)

    u_vectors = []
    for k in range(6):
        theta = oangle_rad + math.radians(30 + 60 * k)
        u_vectors.append((math.cos(theta), math.sin(theta)))

    P_vals = np.zeros(6)
    Q_vals = np.zeros(6)
    for k in range(6):
        ux, uy = u_vectors[k]
        projections = pts[:, 0] * ux + pts[:, 1] * uy
        P_vals[k] = np.max(projections)
        Q_vals[k] = np.min(projections)

    A_mat = np.zeros((12, 3))
    b_vec = np.zeros(12)
    for k in range(6):
        ux, uy = u_vectors[k]
        A_mat[2 * k, 0] = ux
        A_mat[2 * k, 1] = uy
        A_mat[2 * k, 2] = -1
        b_vec[2 * k] = Q_vals[k]

        A_mat[2 * k + 1, 0] = -ux
        A_mat[2 * k + 1, 1] = -uy
        A_mat[2 * k + 1, 2] = -1
        b_vec[2 * k + 1] = -P_vals[k]

    c_vec = np.array([0.0, 0.0, 1.0])

    try:
        result = linprog(c_vec, A_ub=A_mat, b_ub=b_vec,
                         bounds=[(None, None), (None, None), (0, None)],
                         method='highs')
        if result.success:
            A_opt = result.x[2]
            ox_opt = result.x[0]
            oy_opt = result.x[1]
            side = A_opt * 2.0 / SQRT3
            return side, ox_opt, oy_opt
    except Exception:
        pass

    return float('inf'), 0.0, 0.0


def _soft_max(vals, tau):
    """Log-sum-exp soft max."""
    v = np.asarray(vals, dtype=float)
    if v.size == 0:
        return 0.0
    m = v.max()
    return m + tau * math.log(np.sum(np.exp((v - m) / tau)))


def _smooth_enclosing_cost(x, hex_shapes, angle_rad, tau=0.03):
    """
    Smooth surrogate for enclosing hexagon side length.
    x: [x0,y0,...,x10,y10, ox, oy] (24 elements)
    hex_shapes: list of (s, angle_deg) for each inner hex
    """
    n = 11
    pos = x[:2 * n].reshape(n, 2)
    ox = x[2 * n]
    oy = x[2 * n + 1]

    all_pts = []
    for i in range(n):
        s, ang = hex_shapes[i]
        verts = _hex_vertices(pos[i, 0], pos[i, 1], s, ang)
        all_pts.extend(verts)
    pts = np.array(all_pts)

    u = np.zeros((6, 2))
    for k in range(6):
        theta = angle_rad + math.radians(30 + 60 * k)
        u[k, 0] = math.cos(theta)
        u[k, 1] = math.sin(theta)

    proj = pts @ u.T
    P = proj.max(axis=1)
    Q = proj.min(axis=1)

    vals = np.zeros(12)
    for k in range(6):
        vals[2 * k] = ox * u[k, 0] + oy * u[k, 1] - Q[k]
        vals[2 * k + 1] = P[k] - (ox * u[k, 0] + oy * u[k, 1])

    A_soft = _soft_max(vals, tau)
    side = A_soft * 2.0 / SQRT3
    return side


def _separation_penalty(pos, hex_shapes, sep_tau=0.05):
    """
    Smooth penalty for overlaps between inner hexagons.
    Uses pairwise distance minus sum of inradii as a proxy.
    """
    n = 11
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            d = math.sqrt(dx * dx + dy * dy)
            # Minimum center distance for disjoint hexagons (flat-to-flat)
            # For side-1 hexagons, the minimum center distance is SQRT3
            min_dist = SQRT3  # inradius * 2 for flat-to-flat contact
            gap = d - min_dist
            if gap < 0:
                penalty += gap * gap
    return penalty


def _combined_smooth_cost(x, hex_shapes, angle_rad, tau=0.03, weight=50.0):
    """Combined smooth cost: enclosing side + separation penalty."""
    n = 11
    pos = x[:2 * n].reshape(n, 2)
    enc_cost = _smooth_enclosing_cost(x, hex_shapes, angle_rad, tau)
    sep_pen = _separation_penalty(pos, hex_shapes)
    return enc_cost + weight * sep_pen


def construct():
    start_time = time.time()

    def axial_to_xy(q, r):
        x = 1.5 * q
        y = SQRT3 * (r + q / 2.0)
        return (x, y)

    # Center + Ring 1 (6 neighbors)
    pos_axial = [(0, 0)]
    pos_axial.extend([(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)])

    # Ring 2 candidates (12 positions)
    ring2_candidates = [
        (2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
        (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
        (0, 2), (1, 2)
    ]

    best_side = float('inf')
    best_angle = 0.0
    best_center = (0.0, 0.0)
    best_xy = None
    best_combo = None

    all_axial = pos_axial + ring2_candidates
    all_xy = [axial_to_xy(q, r) for q, r in all_axial]
    ring2_indices = list(range(7, len(all_xy)))

    # Iterate over combinations of 4 ring-2 positions
    for combo in itertools.combinations(ring2_indices, 4):
        if time.time() > TIME_LIMIT - 30:
            break

        selected_indices = list(range(7)) + list(combo)
        full_xy = [all_xy[i] for i in selected_indices]

        # Center the cluster at origin
        cx_mean = sum(x for x, y in full_xy) / 11.0
        cy_mean = sum(y for x, y in full_xy) / 11.0
        centered_xy = [(x - cx_mean, y - cy_mean) for x, y in full_xy]
        inner_hexes_centered = [(x, y, 1.0, 0.0) for x, y in centered_xy]

        # Ternary search for best outer rotation in [0, 30] degrees
        lo, hi = 0.0, 30.0
        gr = (math.sqrt(5) - 1) / 2
        c = hi - gr * (hi - lo)
        d = lo + gr * (hi - lo)
        fc = _min_outer_side_analytical(inner_hexes_centered, math.radians(c))[0]
        fd = _min_outer_side_analytical(inner_hexes_centered, math.radians(d))[0]

        for _ in range(30):
            if time.time() > TIME_LIMIT - 30:
                break
            if fc < fd:
                hi = d
                d = c
                fd = fc
                c = hi - gr * (hi - lo)
                fc = _min_outer_side_analytical(inner_hexes_centered, math.radians(c))[0]
            else:
                lo = c
                c = d
                fc = fd
                d = lo + gr * (hi - lo)
                fd = _min_outer_side_analytical(inner_hexes_centered, math.radians(d))[0]

        best_angle_candidate = (lo + hi) / 2.0
        side, ox, oy = _min_outer_side_analytical(
            inner_hexes_centered, math.radians(best_angle_candidate))

        if side < best_side:
            best_side = side
            best_angle = best_angle_candidate
            best_center = (ox + cx_mean, oy + cy_mean)
            best_xy = full_xy
            best_combo = combo

    if best_xy is None:
        # Fallback
        best_xy = [all_xy[i] for i in range(7)]
        best_side = 2.0
        best_angle = 0.0
        best_center = (0.0, 0.0)
        best_combo = tuple(range(7, 11))

    # Build inner hexagons
    inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_xy]

    # Polish phase: fine angle search
    if time.time() < TIME_LIMIT - 20:
        angle_lo = max(0.0, best_angle - 1.0)
        angle_hi = min(30.0, best_angle + 1.0)
        gr = (math.sqrt(5) - 1) / 2
        c = angle_hi - gr * (angle_hi - angle_lo)
        d = angle_lo + gr * (angle_hi - angle_lo)
        fc = _min_outer_side_analytical(inner_hexes, math.radians(c))[0]
        fd = _min_outer_side_analytical(inner_hexes, math.radians(d))[0]
        for _ in range(40):
            if time.time() > TIME_LIMIT - 20:
                break
            if fc < fd:
                angle_hi = d
                d = c
                fd = fc
                c = angle_hi - gr * (angle_hi - angle_lo)
                fc = _min_outer_side_analytical(inner_hexes, math.radians(c))[0]
            else:
                angle_lo = c
                c = d
                fc = fd
                d = angle_lo + gr * (angle_hi - angle_lo)
                fd = _min_outer_side_analytical(inner_hexes, math.radians(d))[0]

        polish_angle = (angle_lo + angle_hi) / 2.0
        side_p, ox_p, oy_p = _min_outer_side_analytical(
            inner_hexes, math.radians(polish_angle))
        if side_p < best_side - 1e-12:
            best_side = side_p
            best_angle = polish_angle
            cx_mean = sum(h[0] for h in inner_hexes) / 11.0
            cy_mean = sum(h[1] for h in inner_hexes) / 11.0
            best_center = (ox_p + cx_mean, oy_p + cy_mean)

    # Smooth refinement: optimize positions using log-sum-exp surrogate
    if time.time() < TIME_LIMIT - 15:
        from scipy.optimize import minimize

        n = 11
        pos0 = np.array([[h[0], h[1]] for h in inner_hexes])
        ox0 = best_center[0] - sum(pos0[:, 0]) / n
        oy0 = best_center[1] - sum(pos0[:, 1]) / n
        x0 = np.concatenate([pos0.flatten(), [ox0, oy0]])
        hex_shapes = [(1.0, 0.0)] * n
        angle_rad = math.radians(best_angle)

        best_smooth_side = best_side
        best_smooth_x = x0.copy()

        for tau in [0.05, 0.02, 0.01]:
            if time.time() > TIME_LIMIT - 15:
                break
            for attempt in range(3):
                if time.time() > TIME_LIMIT - 15:
                    break
                x_init = x0.copy()
                if attempt > 0:
                    rng = np.random.default_rng(hash((tau, attempt)) % (2 ** 32))
                    x_init = x_init + rng.normal(0, 0.01, size=x_init.shape)

                try:
                    res = minimize(
                        _combined_smooth_cost,
                        x_init,
                        args=(hex_shapes, angle_rad, tau, 50.0),
                        method='L-BFGS-B',
                        options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    if res.success:
                        pos_sol = res.x[:2 * n].reshape(n, 2)
                        hexes_sol = [(pos_sol[i, 0], pos_sol[i, 1], 1.0, 0.0)
                                     for i in range(n)]
                        if _all_disjoint(hexes_sol):
                            exact_side, ox_exact, oy_exact = _min_outer_side_analytical(
                                hexes_sol, angle_rad)
                            if exact_side < best_smooth_side - 1e-12:
                                best_smooth_side = exact_side
                                best_smooth_x = res.x.copy()
                except Exception:
                    pass

        if best_smooth_side < best_side - 1e-12:
            pos_final = best_smooth_x[:2 * n].reshape(n, 2)
            ox_final = best_smooth_x[2 * n]
            oy_final = best_smooth_x[2 * n + 1]
            hexes_final = [(pos_final[i, 0], pos_final[i, 1], 1.0, 0.0)
                           for i in range(n)]
            cx_mean = sum(h[0] for h in hexes_final) / n
            cy_mean = sum(h[1] for h in hexes_final) / n
            hexes_centered = [(h[0] - cx_mean, h[1] - cy_mean, 1.0, 0.0)
                              for h in hexes_final]
            best_side, ox, oy = _min_outer_side_analytical(
                hexes_centered, math.radians(best_angle))
            best_center = (ox + cx_mean, oy + cy_mean)
            inner_hexes = [(h[0] + cx_mean, h[1] + cy_mean, 1.0, 0.0)
                           for h in hexes_final]

    # Verify disjointness
    if not _all_disjoint(inner_hexes):
        scale = 1.01
        inner_hexes = [(x * scale, y * scale, 1.0, 0.0) for x, y, s, a in inner_hexes]
        cx_mean = sum(h[0] for h in inner_hexes) / 11.0
        cy_mean = sum(h[1] for h in inner_hexes) / 11.0
        centered = [(h[0] - cx_mean, h[1] - cy_mean, 1.0, 0.0) for h in inner_hexes]
        best_side, ox, oy = _min_outer_side_analytical(
            centered, math.radians(best_angle))
        best_center = (ox + cx_mean, oy + cy_mean)

    # Verify containment
    outer_hex = (best_center[0], best_center[1], best_side, best_angle)
    valid = all(_inner_in_outer(h, outer_hex) for h in inner_hexes)
    if not valid:
        best_side *= 1.005

    inner = [[h[0], h[1], h[3]] for h in inner_hexes]
    return inner, (best_center[0], best_center[1]), best_side, best_angle
