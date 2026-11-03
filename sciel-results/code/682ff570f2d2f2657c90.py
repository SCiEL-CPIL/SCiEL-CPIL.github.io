"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 120.0  # 120s safety margin within 150s limit

# Edge-to-edge center distance for unit regular hexagons (inradius = sqrt(3)/2)
_EDGE_DIST = math.sqrt(3)
# Rest length for the spring system: slightly larger than edge distance
# so that at equilibrium the hexagons are strictly disjoint (tiny gap).
_REST = _EDGE_DIST * 1.0005


def _get_hex_vertices(cx, cy, angle_deg=0.0):
    """Vertices of a unit regular hexagon (circumradius 1) centered at (cx, cy)."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts


def _hex_overlap(c1x, c1y, a1, c2x, c2y, a2, gap=0.0):
    """Exact convex-polygon intersection test for two unit hexagons.
    Returns True if they overlap (or are closer than `gap`), else False.
    Uses the separating-axis theorem (SAT) over the 12 edge normals."""
    dx = c2x - c1x
    dy = c2y - c1y
    d2 = dx * dx + dy * dy
    # Max distance at which two unit hexagons (circumradius 1) can touch = 2
    if d2 > (2.0 + gap) ** 2:
        return False

    v1 = _get_hex_vertices(c1x, c1y, a1)
    v2 = _get_hex_vertices(c2x, c2y, a2)
    for verts in (v1, v2):
        for k in range(6):
            ax, ay = verts[k]
            bx, by = verts[(k + 1) % 6]
            # Edge normal (perpendicular to the edge direction)
            ex, ey = bx - ax, by - ay
            nx, ny = -ey, ex
            # Project both polygons onto this normal
            mn1, mx1 = 1e300, -1e300
            for px, py in v1:
                p = px * nx + py * ny
                if p < mn1:
                    mn1 = p
                if p > mx1:
                    mx1 = p
            mn2, mx2 = 1e300, -1e300
            for px, py in v2:
                p = px * nx + py * ny
                if p < mn2:
                    mn2 = p
                if p > mx2:
                    mx2 = p
            # Separation if the intervals don't overlap (allowing `gap`)
            if mx1 + gap < mn2 or mx2 + gap < mn1:
                return False
    return True


def _build_positions(edges, scale, seed):
    """Place hexagons on a honeycomb lattice: node i at (i*0, i*1)*scale,
    then rotate the whole cluster by a seeded angle and center it."""
    n = len(edges)
    base = [(i * 0.0, i * 1.0) for i in range(n)]
    ang = (seed % 360) * (math.pi / 180.0)
    ca, sa = math.cos(ang), math.sin(ang)
    xs = [x * scale for x, y in base]
    ys = [y * scale for x, y in base]
    # Rotate
    rx = [x * ca - y * sa for x, y in zip(xs, ys)]
    ry = [x * sa + y * ca for x, y in zip(xs, ys)]
    # Center
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    return [[rx[i] - mx, ry[i] - my, 0.0] for i in range(n)]


def _spring_energy(pos, edges, k_spring, k_repel, d0):
    """Energy = sum over edges of (dist - d0)^2 + repulsion for close pairs."""
    n = len(pos)
    E = 0.0
    for i, j in edges:
        dx = pos[j][0] - pos[i][0]
        dy = pos[j][1] - pos[i][1]
        d = math.sqrt(dx * dx + dy * dy)
        E += k_spring * (d - d0) ** 2
    # Repulsion to push apart any pair closer than d0
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[j][0] - pos[i][0]
            dy = pos[j][1] - pos[i][1]
            d2 = dx * dx + dy * dy
            d = math.sqrt(d2) if d2 > 0 else 1e-9
            if d < d0:
                E += k_repel * (d0 - d) ** 2
    return E


def _spring_refine(pos, edges, d0, k_spring, k_repel, iters, lr):
    """Gradient descent on the spring-mass energy to relax the cluster."""
    n = len(pos)
    pos = [[x, y, a] for x, y, a in pos]
    for it in range(iters):
        # Compute forces on each node
        fx = [0.0] * n
        fy = [0.0] * n
        for i, j in edges:
            dx = pos[j][0] - pos[i][0]
            dy = pos[j][1] - pos[i][1]
            d = math.sqrt(dx * dx + dy * dy)
            if d < 1e-9:
                d = 1e-9
            f = k_spring * (d - d0)  # spring force magnitude (toward rest)
            fx[i] += f * dx / d
            fy[i] += f * dy / d
            fx[j] -= f * dx / d
            fy[j] -= f * dy / d
        # Repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[j][0] - pos[i][0]
                dy = pos[j][1] - pos[i][1]
                d2 = dx * dx + dy * dy
                d = math.sqrt(d2) if d2 > 0 else 1e-9
                if d < d0:
                    f = k_repel * (d0 - d)
                    fx[i] -= f * dx / d
                    fy[i] -= f * dy / d
                    fx[j] += f * dx / d
                    fy[j] += f * dy / d
        # Update with adaptive step
        for i in range(n):
            pos[i][0] += lr * fx[i]
            pos[i][1] += lr * fy[i]
        # Decay learning rate
        lr *= 0.995
    return pos


def _min_enclosing_hex_side_lp(inner_hexes, angle_deg):
    """For a fixed outer hexagon angle, find optimal center and side length via LP.
    Returns (cx, cy, side, angle_deg)."""
    all_vertices = []
    for ix, iy, iangle in inner_hexes:
        all_vertices.extend(_get_hex_vertices(ix, iy, iangle))
    n_pts = len(all_vertices)
    if n_pts == 0:
        return 0.0, 0.0, 0.0, angle_deg

    A_ub = []
    b_ub = []
    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        for vx, vy in all_vertices:
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-vx * nx - vy * ny)
    A_ub.append([0.0, 0.0, -1.0])  # r >= 0
    b_ub.append(0.0)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    c = np.array([0.0, 0.0, 1.0])

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
    if result.success:
        cx_opt, cy_opt, r_opt = result.x[0], result.x[1], result.x[2]
    else:
        cx_mean = sum(v[0] for v in all_vertices) / n_pts
        cy_mean = sum(v[1] for v in all_vertices) / n_pts
        r_opt = 0.0
        for k in range(6):
            a = math.radians(angle_deg + 30 + 60 * k)
            nx, ny = math.cos(a), math.sin(a)
            for vx, vy in all_vertices:
                r_opt = max(r_opt, (vx - cx_mean) * nx + (vy - cy_mean) * ny)
        cx_opt, cy_opt = cx_mean, cy_mean

    side = r_opt / (math.sqrt(3) / 2)
    return cx_opt, cy_opt, side, angle_deg


def _min_enclosing_hex_side(inner_hexes):
    """Minimize the outer hexagon side length over center and angle.
    Multi-resolution angle sweep with LP, then Nelder-Mead polish."""
    all_vertices = []
    for ix, iy, iangle in inner_hexes:
        all_vertices.extend(_get_hex_vertices(ix, iy, iangle))
    n_pts = len(all_vertices)
    if n_pts == 0:
        return 0.0, 0.0, 0.0, 0.0

    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # Coarse sweep
    for angle_deg in np.arange(0, 60, 1.0):
        cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg))
        if side < best_side:
            best_side, best_center, best_angle = side, (cx, cy), float(angle_deg)

    # Fine sweep
    for angle_deg in np.arange(best_angle - 3, best_angle + 3, 0.1):
        if angle_deg < 0 or angle_deg >= 60:
            continue
        cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg))
        if side < best_side:
            best_side, best_center, best_angle = side, (cx, cy), float(angle_deg)

    # Finer sweep
    for angle_deg in np.arange(best_angle - 0.5, best_angle + 0.5, 0.01):
        if angle_deg < 0 or angle_deg >= 60:
            continue
        cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg))
        if side < best_side:
            best_side, best_center, best_angle = side, (cx, cy), float(angle_deg)

    # Nelder-Mead polish on (cx, cy, angle)
    if best_side < float('inf') and time.time() < _DEADLINE - 15.0:
        def objective(params):
            cx, cy, angle_deg = params
            angle_deg = angle_deg % 60.0
            r_max = 0.0
            for k in range(6):
                a = math.radians(angle_deg + 30 + 60 * k)
                nx, ny = math.cos(a), math.sin(a)
                for vx, vy in all_vertices:
                    proj = (vx - cx) * nx + (vy - cy) * ny
                    if proj > r_max:
                        r_max = proj
            return r_max / (math.sqrt(3) / 2)

        x0 = np.array([best_center[0], best_center[1], best_angle])
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-8,
                                    'fatol': 1e-10, 'adaptive': True})
            if res.success and res.fun < best_side:
                cx_opt, cy_opt, ang_opt = res.x
                ang_opt = ang_opt % 60.0
                best_side, best_center, best_angle = res.fun, (cx_opt, cy_opt), ang_opt
        except Exception:
            pass

        # Ultra-fine sweep around polished angle
        if best_side < float('inf') and time.time() < _DEADLINE - 10.0:
            for angle_deg in np.arange(best_angle - 0.1, best_angle + 0.1, 0.001):
                if angle_deg < 0 or angle_deg >= 60:
                    continue
                cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg))
                if side < best_side:
                    best_side, best_center, best_angle = side, (cx, cy), float(angle_deg)

    return best_center[0], best_center[1], best_side, best_angle


def _valid(inner, gap=0.001):
    """Check that all pairs are disjoint (with a small safety gap)."""
    for i in range(len(inner)):
        for j in range(i + 1, len(inner)):
            if _hex_overlap(inner[i][0], inner[i][1], inner[i][2],
                            inner[j][0], inner[j][1], inner[j][2], gap):
                return False
    return True


def construct():
    global _START_TIME, _DEADLINE

    # Contact graph: honeycomb cluster = 1 center + 6 first ring + 4 second ring.
    # Nodes: 0 = center; 1..6 = first ring; 7..10 = second ring.
    # Edges: center to all first ring; first ring adjacent pairs;
    # second ring nodes connect to their two nearest first-ring neighbors.
    edges = []
    # Center to first ring
    for j in range(1, 7):
        edges.append((0, j))
    # First ring adjacency (ring of 6)
    for j in range(1, 7):
        edges.append((j, j % 6 + 1))
    # Second ring: place 4 nodes at gaps between first-ring neighbors.
    # Node 7 between 1&2, 8 between 2&3, 9 between 3&4, 10 between 4&5
    for s, (a, b) in enumerate([(1, 2), (2, 3), (3, 4), (4, 5)]):
        edges.append((7 + s, a))
        edges.append((7 + s, b))

    best = None
    best_score = float('inf')

    # Try several initial cluster orientations and scales; relax each with springs.
    scales = [1.0, 1.002, 1.004]
    seeds = [0, 15, 30, 45, 60, 75, 90, 120, 150, 180, 210, 240, 270, 300, 330]

    for seed in seeds:
        if time.time() > _DEADLINE - 25.0:
            break
        for scale in scales:
            pos = _build_positions(edges, scale, seed)
            # Spring relaxation
            pos = _spring_refine(pos, edges, _REST, k_spring=1.0,
                                 k_repel=50.0, iters=300, lr=0.05)
            if not _valid(pos, gap=0.002):
                continue
            # Compute enclosing hexagon
            cx, cy, side, angle = _min_enclosing_hex_side(pos)
            if side < best_score:
                best_score = side
                best = (pos, cx, cy, side, angle)

    # Fallback if nothing valid found
    if best is None:
        # Spread hexagons far apart in a line (always valid)
        inner = [[10.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 60.0, 0.0

    pos, cx, cy, side, angle = best

    # Final safety: if overlaps somehow remain, fall back to a safe spread.
    if not _valid(pos, gap=0.0):
        inner = [[10.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 60.0, 0.0

    return pos, (cx, cy), side, angle
# EVOLVE-BLOCK-END
