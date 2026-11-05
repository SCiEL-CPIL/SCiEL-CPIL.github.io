import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)
EPS = 1e-9

def _hex_vertices(cx, cy, angle_deg, s=1.0):
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([s * np.cos(angles), s * np.sin(angles)])
    return vs + np.array([cx, cy])

def _outer_hex_vertices(cx, cy, side, angle_deg):
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([side * np.cos(angles), side * np.sin(angles)])
    return vs + np.array([cx, cy])

def _point_in_polygon(pt, poly):
    n = len(poly)
    sign = 0
    for i in range(n):
        a = poly[i]
        b = poly[(i+1) % n]
        cross = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
        if cross > 1e-12:
            if sign < 0:
                return False
            sign = 1
        elif cross < -1e-12:
            if sign > 0:
                return False
            sign = -1
    return True

def _seg_seg_dist(p1, p2, p3, p4):
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    if a < 1e-12 and e < 1e-12:
        return np.linalg.norm(p1 - p3)
    if a < 1e-12:
        s = 0
        t = f / e
        t = max(0, min(1, t))
        d = p1 - (p3 + t * d2)
        return np.linalg.norm(d)
    c = np.dot(d1, r)
    if e < 1e-12:
        t = 0
        s = c / a
        s = max(0, min(1, s))
        d = (p1 + s * d1) - p3
        return np.linalg.norm(d)
    b = np.dot(d1, d2)
    denom = a * e - b * b
    s = 0
    t = 0
    if denom > 1e-12:
        s = (b * f - c * e) / denom
        s = max(0, min(1, s))
        t = (b * s + f) / e
        if t < 0:
            t = 0
            s = max(0, min(1, c / a))
        elif t > 1:
            t = 1
            s = max(0, min(1, (b + f) / a))
        d = (p1 + s * d1) - (p3 + t * d2)
        return np.linalg.norm(d)
    else:
        s = 0
        t = f / e
        t = max(0, min(1, t))
        d = p1 - (p3 + t * d2)
        return np.linalg.norm(d)

def _poly_separation(p1, p2):
    for v in p1:
        if _point_in_polygon(v, p2):
            return 0.0
    for v in p2:
        if _point_in_polygon(v, p1):
            return 0.0
    min_dist = float('inf')
    n1 = len(p1)
    n2 = len(p2)
    for i in range(n1):
        a1 = p1[i]
        a2 = p1[(i+1) % n1]
        for j in range(n2):
            b1 = p2[j]
            b2 = p2[(j+1) % n2]
            d = _seg_seg_dist(a1, a2, b1, b2)
            if d < min_dist:
                min_dist = d
    return min_dist

def _compute_side(inner_params, outer_angle_deg):
    """Binary search for minimum outer side containing all inner hexagons."""
    n = len(inner_params)
    lo = 2.0
    hi = 12.0
    for _ in range(50):
        mid = (lo + hi) / 2
        outer_v = _outer_hex_vertices(0.0, 0.0, mid, outer_angle_deg)
        valid = True
        for i in range(n):
            x = inner_params[i, 0]
            y = inner_params[i, 1]
            a = inner_params[i, 2]
            hex_v = _hex_vertices(x, y, a)
            for v in hex_v:
                if not _point_in_polygon(v, outer_v):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            hi = mid
        else:
            lo = mid
    return hi

def _overlap_penalty(inner_params, penalty_weight=200.0):
    n = len(inner_params)
    penalty = 0.0
    for i in range(n):
        x1 = inner_params[i, 0]
        y1 = inner_params[i, 1]
        a1 = inner_params[i, 2]
        v1 = _hex_vertices(x1, y1, a1)
        for j in range(i+1, n):
            x2 = inner_params[j, 0]
            y2 = inner_params[j, 1]
            a2 = inner_params[j, 2]
            v2 = _hex_vertices(x2, y2, a2)
            sep = _poly_separation(v1, v2)
            if sep < 0.01:
                penalty += penalty_weight * (0.01 - sep) ** 2
    return penalty

def _objective(params, outer_angle_deg, n):
    inner_params = params.reshape(n, 3)
    side = _compute_side(inner_params, outer_angle_deg)
    overlap_pen = _overlap_penalty(inner_params)
    return side + overlap_pen

def _is_valid(inner_params, side, outer_angle_deg, min_sep=0.0):
    n = len(inner_params)
    outer_v = _outer_hex_vertices(0.0, 0.0, side, outer_angle_deg)
    for i in range(n):
        x = inner_params[i, 0]
        y = inner_params[i, 1]
        a = inner_params[i, 2]
        hex_v = _hex_vertices(x, y, a)
        for v in hex_v:
            if not _point_in_polygon(v, outer_v):
                return False
    for i in range(n):
        x1 = inner_params[i, 0]
        y1 = inner_params[i, 1]
        a1 = inner_params[i, 2]
        v1 = _hex_vertices(x1, y1, a1)
        for j in range(i+1, n):
            x2 = inner_params[j, 0]
            y2 = inner_params[j, 1]
            a2 = inner_params[j, 2]
            v2 = _hex_vertices(x2, y2, a2)
            sep = _poly_separation(v1, v2)
            if sep < min_sep:
                return False
    return True

def _seed_solution():
    """Concentric ring: 1 center + 6 at dist √3 + 4 at dist 3."""
    eps = 1e-9
    centers = []
    centers.append((0.0, 0.0))
    for k in range(6):
        angle = np.deg2rad(k * 60)
        x = (SQRT3 + eps) * np.cos(angle)
        y = (SQRT3 + eps) * np.sin(angle)
        centers.append((x, y))
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        x = (3.0 + 2*eps) * np.cos(angle)
        y = (3.0 + 2*eps) * np.sin(angle)
        centers.append((x, y))
    inner = []
    for x, y in centers:
        inner.append([x, y, 30.0])
    max_x = (2 + eps) * SQRT3
    max_y = 4 + 2 * eps
    s_x = 2 * max_x / SQRT3
    s_y = max_y
    side = max(s_x, s_y) + 1e-9
    return inner, (0.0, 0.0), side, 30.0

def _build_config(outer_angles, outer_r, inner_r=SQRT3 + EPS, outer_angle_deg=30.0, inner_angle_deg=30.0):
    """
    Build configuration: center + 6 inner ring at inner_r + 4 outer ring at outer_r.
    outer_angles: list of 4 angles (degrees) for outer hexagons.
    """
    params = np.zeros((11, 3))
    # Center
    params[0] = [0.0, 0.0, inner_angle_deg]
    # 6 inner ring
    for k in range(6):
        angle_rad = np.deg2rad(k * 60)
        params[1 + k, 0] = inner_r * np.cos(angle_rad)
        params[1 + k, 1] = inner_r * np.sin(angle_rad)
        params[1 + k, 2] = inner_angle_deg
    # 4 outer ring
    for idx, ang_deg in enumerate(outer_angles):
        angle_rad = np.deg2rad(ang_deg)
        params[7 + idx, 0] = outer_r * np.cos(angle_rad)
        params[7 + idx, 1] = outer_r * np.sin(angle_rad)
        params[7 + idx, 2] = outer_angle_deg
    return params

def _local_refine(params, outer_angle_deg, n, maxiter=400):
    try:
        res = minimize(_objective, params.flatten(), args=(outer_angle_deg, n),
                      method='Nelder-Mead',
                      options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8})
        return res.x.reshape(n, 3), res.fun
    except Exception:
        return params, _objective(params.flatten(), outer_angle_deg, n)

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    n = 11

    # Try multiple outer hexagon orientations
    best_inner = None
    best_side = float('inf')
    best_outer_angle = 30.0

    # Outer hexagon orientations to try
    outer_angles_to_try = [0.0, 30.0, 15.0, 45.0, 60.0, -15.0, -30.0]

    # The 6 pointy positions (angles in degrees)
    pointy_positions = [30, 90, 150, 210, 270, 330]
    # Choose 4 of 6: all combinations
    from itertools import combinations
    all_combos = list(combinations(range(6), 4))

    for outer_angle in outer_angles_to_try:
        if time.time() - start_time > DEADLINE - 10:
            break

        for combo in all_combos:
            if time.time() - start_time > DEADLINE - 10:
                break

            angles_deg = [pointy_positions[i] for i in combo]
            params = _build_config(angles_deg, 3.0 + 2*EPS)

            # Quick local refine
            refined, val = _local_refine(params, outer_angle, n, maxiter=200)
            side = _compute_side(refined, outer_angle)
            if _is_valid(refined, side, outer_angle, min_sep=1e-9):
                if side < best_side:
                    best_side = side
                    best_inner = refined.copy()
                    best_outer_angle = outer_angle

    # If we found something, do more refinement
    if best_inner is not None:
        # More thorough local refinement
        remaining = DEADLINE - 15 - (time.time() - start_time)
        if remaining > 5:
            refined, val = _local_refine(best_inner, best_outer_angle, n, maxiter=500)
            side = _compute_side(refined, best_outer_angle)
            if _is_valid(refined, side, best_outer_angle, min_sep=1e-9):
                if side < best_side:
                    best_side = side
                    best_inner = refined

    # Fallback to seed if nothing found
    if best_inner is None:
        seed_inner, seed_center, seed_side, seed_angle = _seed_solution()
        best_inner = np.array(seed_inner)
        best_side = seed_side
        best_outer_angle = seed_angle

    # Final validation
    final_side = _compute_side(best_inner, best_outer_angle)
    if not _is_valid(best_inner, final_side, best_outer_angle, min_sep=1e-9):
        seed_inner, seed_center, seed_side, seed_angle = _seed_solution()
        best_inner = np.array(seed_inner)
        final_side = seed_side
        best_outer_angle = seed_angle

    inner_rows = []
    for row in best_inner:
        inner_rows.append([float(row[0]), float(row[1]), float(row[2])])

    center = (0.0, 0.0)
    side = float(final_side)
    angle_deg = float(best_outer_angle)

    return inner_rows, center, side, angle_deg
