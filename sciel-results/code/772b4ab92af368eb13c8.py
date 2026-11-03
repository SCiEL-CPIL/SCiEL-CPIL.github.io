"""
Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Uses a robust multi-topology approach:
1. Parameterized lattice structures (2x2x3 grid, hexagonal clusters, rings).
2. Local optimization of lattice parameters (spacing, rotation).
3. Free-placement refinement with hard-constraint enforcement via SLSQP.
4. High-precision Nelder-Mead polish.
"""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = time.time() + 148.0  # 2s safety margin before 150s kill


def hex_vertices(cx, cy, s, angle_deg):
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
    return verts


def point_in_convex_polygon(px, py, poly):
    n = len(poly)
    sign = 0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if abs(cross) < 1e-10:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


def sat_gap(A, B):
    """Maximum SAT gap: positive = separated, negative = overlap depth."""
    max_gap = -1e10
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            dx, dy = x2 - x1, y2 - y1
            nx, ny = -dy, dx
            norm = math.sqrt(nx * nx + ny * ny)
            if norm < 1e-12:
                continue
            nx, ny = nx / norm, ny / norm
            minA = min(nx * px + ny * py for px, py in A)
            maxA = max(nx * px + ny * py for px, py in A)
            minB = min(nx * px + ny * py for px, py in B)
            maxB = max(nx * px + ny * py for px, py in B)
            if maxA < minB:
                gap = minB - maxA
            elif maxB < minA:
                gap = minA - maxB
            else:
                gap = -(min(maxA, maxB) - max(minA, minB))
            if gap > max_gap:
                max_gap = gap
    return max_gap


def polygons_overlap_sat(A, B):
    return sat_gap(A, B) < -1e-9


def compute_min_outer_side(centers, angles, ocx, ocy, otheta_deg):
    otheta = math.radians(otheta_deg)
    max_proj = 0.0
    for k in range(6):
        normal_angle = otheta + math.pi / 6 + k * math.pi / 3
        nx = math.cos(normal_angle)
        ny = math.sin(normal_angle)
        for i in range(len(centers)):
            cx, cy = centers[i]
            angle = math.radians(angles[i])
            for vi in range(6):
                vert_angle = angle + vi * math.pi / 3
                vx = cx + math.cos(vert_angle)
                vy = cy + math.sin(vert_angle)
                proj = (vx - ocx) * nx + (vy - ocy) * ny
                if proj > max_proj:
                    max_proj = proj
    return 2.0 * max_proj / math.sqrt(3)


def check_validity(centers, angles, ocx, ocy, outer_side, outer_angle):
    outer = hex_vertices(ocx, ocy, outer_side, outer_angle)
    polys = []
    for i in range(len(centers)):
        cx, cy = centers[i]
        ang = angles[i]
        p = hex_vertices(cx, cy, 1.0, ang)
        polys.append(p)
    for p in polys:
        for (px, py) in p:
            if not point_in_convex_polygon(px, py, outer):
                return False
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polygons_overlap_sat(polys[i], polys[j]):
                return False
    return True


def smooth_max(vals, beta=50.0):
    v = np.array(vals, dtype=float)
    v_max = np.max(v)
    if v_max > 1e10:
        return 1e10
    shifted = v - v_max
    return v_max + math.log(np.sum(np.exp(beta * shifted))) / beta


def build_lattice_2x2x3(params):
    """
    2x2x3 grid of hexagons.
    params: [dx, dy, rot]
    dx, dy: spacing between centers
    rot: rotation of the grid
    """
    dx, dy, rot = params
    centers = []
    angles = []
    # 3 rows, 2 columns? Or 2 rows, 3 columns?
    # Let's try 3 rows of 2, with offset.
    # Actually, a 2x2x3 block in hex lattice.
    # Let's use a rectangular grid with hex offset.
    for r in range(3):
        for c in range(2):
            x = c * dx
            y = r * dy
            if c % 2 == 1:
                x += dx / 2.0
            centers.append([x, y])
            angles.append(rot)
    # Rotate the whole set
    c_rot = math.cos(math.radians(rot))
    s_rot = math.sin(math.radians(rot))
    new_centers = []
    for cx, cy in centers:
        nx = cx * c_rot - cy * s_rot
        ny = cx * s_rot + cy * c_rot
        new_centers.append([nx, ny])
    return new_centers, angles


def build_hex_cluster(params):
    """
    Central hex + 5 ring + 6 outer? No, 1+5=6. 1+5+6=12.
    params: [r1, r2, rot]
    """
    r1, r2, rot = params
    centers = [[0.0, 0.0]]
    angles = [0.0]
    # Ring 1: 6 hexagons
    for k in range(6):
        theta = math.radians(rot + k * 60.0)
        centers.append([r1 * math.cos(theta), r1 * math.sin(theta)])
        angles.append(rot)
    # Ring 2: 6 hexagons (offset by 30 deg? or 0?)
    # In a hex lattice, the second ring has 12 positions, but we only need 6.
    # Let's place them at the "corners" of the hexagonal cluster.
    for k in range(6):
        theta = math.radians(rot + k * 60.0 + 30.0)
        # Distance to corner of 1+5 cluster is r1 * sqrt(3) approx?
        # Let's just use r2.
        centers.append([r2 * math.cos(theta), r2 * math.sin(theta)])
        angles.append(rot)
    return centers, angles


def build_ring12(params):
    """
    12 hexagons in a ring.
    params: [r, rot]
    """
    r, rot = params
    centers = []
    angles = []
    for k in range(12):
        theta = math.radians(rot + k * 30.0)
        centers.append([r * math.cos(theta), r * math.sin(theta)])
        angles.append(rot)
    return centers, angles


def build_rect_3x4(params):
    """
    3x4 rectangular grid.
    params: [dx, dy, rot]
    """
    dx, dy, rot = params
    centers = []
    angles = []
    for r in range(3):
        for c in range(4):
            x = (c - 1.5) * dx
            y = (r - 1.0) * dy
            centers.append([x, y])
            angles.append(rot)
    c_rot = math.cos(math.radians(rot))
    s_rot = math.sin(math.radians(rot))
    new_centers = []
    for cx, cy in centers:
        nx = cx * c_rot - cy * s_rot
        ny = cx * s_rot + cy * c_rot
        new_centers.append([nx, ny])
    return new_centers, angles


def objective_lattice(params, build_func):
    if time.time() > DEADLINE - 20:
        return 1e10
    try:
        centers, angles = build_func(params)
    except Exception:
        return 1e10
    
    # Compute min outer side with outer hex centered at (0,0) and rot 0 for now
    side = compute_min_outer_side(centers, angles, 0.0, 0.0, 0.0)
    
    # Overlap penalty
    polys = [hex_vertices(c[0], c[1], 1.0, a) for c, a in zip(centers, angles)]
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            gap = sat_gap(polys[i], polys[j])
            if gap < 0.0:
                penalty += gap ** 2
    return side + 1000.0 * penalty


def polish_outer(centers, angles, ocx, ocy, otheta, max_rounds=6):
    best_side = compute_min_outer_side(centers, angles, ocx, ocy, otheta)
    best_ocx, best_ocy, best_otheta = ocx, ocy, otheta
    for _ in range(max_rounds):
        if time.time() > DEADLINE - 10:
            break
        improved = False
        for scale in [0.05, 0.01, 0.002, 0.0005, 0.0001]:
            for dx in [-scale, 0, scale]:
                for dy in [-scale, 0, scale]:
                    for da in [-scale * 5, 0, scale * 5]:
                        side = compute_min_outer_side(centers, angles,
                                                      best_ocx + dx, best_ocy + dy,
                                                      best_otheta + da)
                        if side < best_side - 1e-12:
                            best_side = side
                            best_ocx += dx
                            best_ocy += dy
                            best_otheta += da
                            improved = True
        if not improved:
            break
    return best_ocx, best_ocy, best_otheta, best_side


def free_placement_objective(x, beta=40.0):
    """Full free-placement objective with smooth overlap penalty."""
    if time.time() > DEADLINE - 15:
        return 1e10
    centers = [[x[3 * i], x[3 * i + 1]] for i in range(N)]
    angles = [x[3 * i + 2] for i in range(N)]
    ocx, ocy, otheta = x[3 * N], x[3 * N + 1], x[3 * N + 2]

    # Smooth outer side
    otheta_r = math.radians(otheta)
    projections = []
    for k in range(6):
        normal_angle = otheta_r + math.pi / 6 + k * math.pi / 3
        nx = math.cos(normal_angle)
        ny = math.sin(normal_angle)
        for i in range(N):
            cx, cy = centers[i]
            angle = math.radians(angles[i])
            for vi in range(6):
                vert_angle = angle + vi * math.pi / 3
                vx = cx + math.cos(vert_angle)
                vy = cy + math.sin(vert_angle)
                proj = (vx - ocx) * nx + (vy - ocy) * ny
                projections.append(proj)
    side = 2.0 * smooth_max(projections, beta) / math.sqrt(3)

    # Smooth overlap penalty
    penalty = 0.0
    polys = [hex_vertices(c[0], c[1], 1.0, a) for c, a in zip(centers, angles)]
    for i in range(N):
        for j in range(i + 1, N):
            gap = sat_gap(polys[i], polys[j])
            if gap < 0.002:
                penalty += (0.002 - gap) ** 2
    return side + 800.0 * penalty


def construct():
    t0 = time.time()
    best_side = float('inf')
    best_result = None

    def try_result(centers, angles, ocx, ocy, otheta):
        nonlocal best_side, best_result
        if time.time() > DEADLINE - 5:
            return
        side = compute_min_outer_side(centers, angles, ocx, ocy, otheta)
        if check_validity(centers, angles, ocx, ocy, side, otheta):
            if side < best_side:
                best_side = side
                best_result = (centers, angles, ocx, ocy, side, otheta)

    # Phase 1: Optimize specific lattice topologies
    topologies = [
        (build_lattice_2x2x3, [2.0, 2.0, 0.0]),
        (build_lattice_2x2x3, [2.0, 2.0, 30.0]),
        (build_hex_cluster, [2.0, 4.0, 0.0]),
        (build_hex_cluster, [2.0, 4.0, 30.0]),
        (build_ring12, [2.5, 0.0]),
        (build_ring12, [2.5, 15.0]),
        (build_rect_3x4, [2.0, 2.0, 0.0]),
        (build_rect_3x4, [2.0, 2.0, 30.0]),
    ]

    for build_func, x0 in topologies:
        if time.time() > DEADLINE - 40:
            break
        try:
            res = minimize(objective_lattice, x0, args=(build_func,),
                           method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < 1e9:
                params = res.x
                centers, angles = build_func(params)
                # Polish outer
                ocx, ocy, otheta, side = polish_outer(centers, angles, 0.0, 0.0, 0.0)
                try_result(centers, angles, ocx, ocy, otheta)
        except Exception:
            pass

    # Phase 2: Free-placement refinement from best topology result
    if best_result is not None and time.time() < DEADLINE - 20:
        c0, a0, ocx0, ocy0, _, otheta0 = best_result
        x0 = []
        for i in range(N):
            x0.extend([c0[i][0], c0[i][1], a0[i]])
        x0.extend([ocx0, ocy0, otheta0])

        bounds = []
        for i in range(N):
            bounds.extend([(-6, 6), (-6, 6), (-30, 30)])
        bounds.extend([(-3, 3), (-3, 3), (-30, 30)])

        try:
            res = minimize(free_placement_objective, x0, method='SLSQP',
                           bounds=bounds, options={'maxiter': 200, 'ftol': 1e-12})
            if res.fun < 1e9:
                x = res.x
                centers = [[x[3 * i], x[3 * i + 1]] for i in range(N)]
                angles = [x[3 * i + 2] for i in range(N)]
                ocx, ocy, otheta = x[3 * N], x[3 * N + 1], x[3 * N + 2]
                ocx, ocy, otheta, side = polish_outer(centers, angles, ocx, ocy, otheta)
                try_result(centers, angles, ocx, ocy, otheta)
        except Exception:
            pass

    # Phase 3: Nelder-Mead free refinement (positions only, fix angles)
    if best_result is not None and time.time() < DEADLINE - 10:
        c0, a0, ocx0, ocy0, _, otheta0 = best_result
        x0 = []
        for i in range(N):
            x0.extend([c0[i][0], c0[i][1]])
        x0.extend([ocx0, ocy0, otheta0])

        def nm_obj(x):
            if time.time() > DEADLINE - 5:
                return 1e10
            centers = [[x[2 * i], x[2 * i + 1]] for i in range(N)]
            angles = [a0[i] for i in range(N)]
            ocx, ocy, otheta = x[2 * N], x[2 * N + 1], x[2 * N + 2]
            side = compute_min_outer_side(centers, angles, ocx, ocy, otheta)
            penalty = 0.0
            polys = [hex_vertices(c[0], c[1], 1.0, a) for c, a in zip(centers, angles)]
            for i in range(N):
                for j in range(i + 1, N):
                    gap = sat_gap(polys[i], polys[j])
                    if gap < 0.001:
                        penalty += (0.001 - gap) ** 2
            return side + 500.0 * penalty

        try:
            res = minimize(nm_obj, x0, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9})
            if res.fun < 1e9:
                x = res.x
                centers = [[x[2 * i], x[2 * i + 1]] for i in range(N)]
                angles = [a0[i] for i in range(N)]
                ocx, ocy, otheta = x[2 * N], x[2 * N + 1], x[2 * N + 2]
                ocx, ocy, otheta, side = polish_outer(centers, angles, ocx, ocy, otheta)
                try_result(centers, angles, ocx, ocy, otheta)
        except Exception:
            pass

    # Fallback
    if best_result is None:
        a = 2.002
        positions = [(0.0, 0.0)]
        for k in range(6):
            theta = math.radians(k * 60.0)
            positions.append((a * math.cos(theta), a * math.sin(theta)))
        for k in range(5):
            theta = math.radians(k * 72.0 + 18.0)
            positions.append((2.0 * a * math.cos(theta), 2.0 * a * math.sin(theta)))
        centers = [list(p) for p in positions]
        angles = [0.0] * 12
        outer_side = compute_min_outer_side(centers, angles, 0.0, 0.0, 0.0)
        inner = [(centers[i][0], centers[i][1], angles[i]) for i in range(N)]
        return inner, (0.0, 0.0), outer_side, 0.0

    c, a, ocx, ocy, side, otheta = best_result
    inner = [(c[i][0], c[i][1], a[i]) for i in range(N)]
    return inner, (ocx, ocy), side, otheta
