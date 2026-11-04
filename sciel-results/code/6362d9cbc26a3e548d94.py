import math
import time
import numpy as np
import random
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

# Precompute inner-hexagon vertex offsets for each integer rotation mod 30 deg
# (hexagon has 60 deg symmetry; 30 deg gives the pointy-top orientation used in the seed)
_INNER_OFFSETS = {}
for _k in range(12):
    _a = math.radians(_k * 30.0)
    _ca, _sa = math.cos(_a), math.sin(_a)
    _rows = []
    for _m in range(6):
        _th = _a + math.radians(60 * _m)
        _rows.append([math.cos(_th), math.sin(_th)])
    _INNER_OFFSETS[_k] = np.array(_rows)

def _rot30_key(angle_deg):
    """Map an angle to its rotation class mod 30 deg (int 0..11)."""
    return int(round(angle_deg / 30.0)) % 12

def _hex_vertices(cx, cy, angle_deg, s=1.0):
    """Vertices of a regular hexagon with side s, centered at (cx,cy), rotated angle_deg."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([s * np.cos(angles), s * np.sin(angles)])
    return vs + np.array([cx, cy])

def _point_in_polygon(pt, poly):
    """Check if point is inside convex polygon using cross products."""
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
    """Minimum distance between two line segments p1-p2 and p3-p4."""
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
    """
    Compute the minimum distance between two convex polygons.
    Returns 0 if they overlap.
    """
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

def _outer_hex_vertices(cx, cy, side, angle_deg):
    """Vertices of outer regular hexagon."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([side * np.cos(angles), side * np.sin(angles)])
    return vs + np.array([cx, cy])

def _all_inner_vertices(inner_params):
    """Vectorized: all 66 vertices of the inner hexagons as (66, 2)."""
    n = len(inner_params)
    verts = np.empty((n * 6, 2))
    for i in range(n):
        key = _rot30_key(inner_params[i, 2])
        off = _INNER_OFFSETS[key]
        verts[i*6:(i+1)*6, 0] = inner_params[i, 0] + off[:, 0]
        verts[i*6:(i+1)*6, 1] = inner_params[i, 1] + off[:, 1]
    return verts

def _outer_normals(angle_deg):
    """Unit outward face normals of the outer hexagon (6, 2)."""
    base = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]) + angle_deg)
    return np.column_stack([np.cos(base), np.sin(base)])

def _min_side_projection(verts, outer_center, outer_angle_deg):
    """Exact minimum outer side length enclosing the point set (single matrix multiply)."""
    normals = _outer_normals(outer_angle_deg)
    rel = verts - outer_center
    dots = rel @ normals.T
    max_proj = np.max(np.abs(dots))
    return 2.0 * max_proj / SQRT3

def _compute_side(inner_params, outer_angle_deg):
    """
    Compute the minimum outer hexagon side length such that all inner hexagons
    are contained within the outer hexagon (outer centered at origin).
    Uses the fast exact projection formula.
    inner_params: array of shape (n, 3) with [x, y, angle_deg]
    """
    verts = _all_inner_vertices(inner_params)
    return _min_side_projection(verts, np.array([0.0, 0.0]), outer_angle_deg)

def _smooth_side(verts, outer_angle_deg, beta=200.0):
    """Smooth (log-sum-exp) surrogate of the enclosing side; differentiable."""
    normals = _outer_normals(outer_angle_deg)
    dots = (verts @ normals.T) / SQRT3
    m = np.max(dots, axis=0)
    s_pos = np.log(np.sum(np.exp(beta * (dots - m)), axis=0) / 6.0) + m
    m2 = np.min(dots, axis=0)
    s_neg = np.log(np.sum(np.exp(-beta * (dots - m2)), axis=0) / 6.0) - m2
    return 2.0 * np.max(np.maximum(s_pos, s_neg))

def _overlap_penalty(inner_params, penalty_weight=200.0):
    """
    Compute penalty for overlapping inner hexagons.
    """
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
    """
    Objective function: outer side length + overlap penalty.
    """
    inner_params = params.reshape(n, 3)
    side = _compute_side(inner_params, outer_angle_deg)
    overlap_pen = _overlap_penalty(inner_params)
    return side + overlap_pen

def _is_valid(inner_params, side, outer_angle_deg, min_sep=0.0):
    """Check if the configuration is valid."""
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
    """Return the concentric ring seed solution."""
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

def _psp_swarm(inner_params, outer_angle_deg, n, iterations=300, swarm_size=30, time_limit=30.0):
    """
    Particle Swarm Optimization to minimize the objective.
    Returns the best particle found.
    """
    start_time = time.time()
    best_global = inner_params.copy()
    best_global_val = _objective(best_global.flatten(), outer_angle_deg, n)

    # Initialize swarm
    particles = np.zeros((swarm_size, n, 3))
    velocities = np.zeros((swarm_size, n, 3))
    personal_bests = np.zeros((swarm_size, n, 3))
    personal_best_vals = np.zeros(swarm_size)

    for i in range(swarm_size):
        if i == 0:
            particles[i] = inner_params
        else:
            particles[i] = inner_params + np.random.randn(n, 3) * 0.5
            particles[i, :, 2] = np.mod(particles[i, :, 2], 360)

        val = _objective(particles[i].flatten(), outer_angle_deg, n)
        personal_bests[i] = particles[i]
        personal_best_vals[i] = val

        if val < best_global_val:
            best_global_val = val
            best_global = particles[i].copy()

    # PSO parameters
    w = 0.9  # inertia
    c1 = 1.5  # cognitive
    c2 = 1.5  # social

    for it in range(iterations):
        if time.time() - start_time > time_limit:
            break

        for i in range(swarm_size):
            r1 = np.random.rand(n, 3)
            r2 = np.random.rand(n, 3)

            velocities[i] = w * velocities[i] + c1 * r1 * (personal_bests[i] - particles[i]) + c2 * r2 * (best_global - particles[i])

            # Clamp velocities
            max_vel = 0.5
            velocities[i] = np.clip(velocities[i], -max_vel, max_vel)

            particles[i] = particles[i] + velocities[i]
            particles[i, :, 2] = np.mod(particles[i, :, 2], 360)

            val = _objective(particles[i].flatten(), outer_angle_deg, n)

            if val < personal_best_vals[i]:
                personal_best_vals[i] = val
                personal_bests[i] = particles[i].copy()

                if val < best_global_val:
                    best_global_val = val
                    best_global = particles[i].copy()

        # Decay inertia
        w = max(0.4, w * 0.999)

    return best_global, best_global_val

def _local_refine(inner_params, outer_angle_deg, n, time_limit=20.0):
    """
    Use Nelder-Mead for local refinement.
    """
    start_time = time.time()

    init_params = inner_params.flatten()

    try:
        res = minimize(_objective, init_params, args=(outer_angle_deg, n),
                      method='Nelder-Mead',
                      options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-7})

        if time.time() - start_time < time_limit:
            candidate = res.x.reshape(n, 3)
            if res.fun < _objective(inner_params.flatten(), outer_angle_deg, n):
                return candidate
    except Exception:
        pass

    return inner_params

def _smooth_polish(inner_params, outer_angle_deg, n, time_limit=15.0):
    """
    Final high-precision polish using the smooth log-sum-exp surrogate with
    an overlap penalty, followed by a tight Nelder-Mead pass on the exact objective.
    """
    start_time = time.time()

    def _smooth_obj(params):
        ip = params.reshape(n, 3)
        verts = _all_inner_vertices(ip)
        return _smooth_side(verts, outer_angle_deg, beta=200.0) + _overlap_penalty(ip, 50.0)

    best = inner_params.copy()
    try:
        res = minimize(_smooth_obj, inner_params.flatten(), method='L-BFGS-B',
                       options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-9})
        if time.time() - start_time < time_limit:
            cand = res.x.reshape(n, 3)
            if _objective(cand.flatten(), outer_angle_deg, n) < _objective(best.flatten(), outer_angle_deg, n):
                best = cand
    except Exception:
        pass

    # Tight exact Nelder-Mead on top
    if time.time() - start_time < time_limit - 2.0:
        try:
            res2 = minimize(_objective, best.flatten(), args=(outer_angle_deg, n),
                            method='Nelder-Mead',
                            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
            cand2 = res2.x.reshape(n, 3)
            if res2.fun < _objective(best.flatten(), outer_angle_deg, n):
                best = cand2
        except Exception:
            pass

    return best

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    safety_margin = 5.0

    n = 11
    outer_angle = 30.0

    # Seed solution
    seed_inner, seed_center, seed_side, seed_angle = _seed_solution()
    best_inner = np.array(seed_inner)
    best_side = _compute_side(best_inner, outer_angle)

    # Initial Nelder-Mead to get a good starting point
    init_params = best_inner.flatten()
    try:
        res = minimize(_objective, init_params, args=(outer_angle, n),
                      method='Nelder-Mead',
                      options={'maxiter': 400, 'xatol': 1e-7, 'fatol': 1e-7})
        if res.fun < best_side:
            candidate = res.x.reshape(n, 3)
            candidate_side = _compute_side(candidate, outer_angle)
            if _is_valid(candidate, candidate_side, outer_angle, min_sep=1e-9):
                best_inner = candidate
                best_side = candidate_side
    except Exception:
        pass

    # PSO phase
    remaining = DEADLINE - safety_margin - (time.time() - start_time)
    if remaining > 10:
        pso_time = min(remaining * 0.5, 60.0)
        try:
            pso_result, pso_val = _psp_swarm(best_inner, outer_angle, n, iterations=400, swarm_size=30, time_limit=pso_time)
            if pso_val < best_side:
                candidate = pso_result
                candidate_side = _compute_side(candidate, outer_angle)
                if _is_valid(candidate, candidate_side, outer_angle, min_sep=1e-9):
                    best_inner = candidate
                    best_side = candidate_side
        except Exception:
            pass

    # Local refinement phase
    remaining = DEADLINE - safety_margin - (time.time() - start_time)
    if remaining > 5:
        try:
            refined = _local_refine(best_inner, outer_angle, n, time_limit=min(remaining * 0.6, 40.0))
            refined_side = _compute_side(refined, outer_angle)
            if refined_side < best_side and _is_valid(refined, refined_side, outer_angle, min_sep=1e-9):
                best_inner = refined
                best_side = refined_side
        except Exception:
            pass

    # Final high-precision smooth polish
    remaining = DEADLINE - safety_margin - (time.time() - start_time)
    if remaining > 4:
        try:
            polished = _smooth_polish(best_inner, outer_angle, n, time_limit=min(remaining * 0.8, 25.0))
            polished_side = _compute_side(polished, outer_angle)
            if polished_side < best_side and _is_valid(polished, polished_side, outer_angle, min_sep=1e-9):
                best_inner = polished
                best_side = polished_side
        except Exception:
            pass

    # Final check
    final_side = _compute_side(best_inner, outer_angle)
    if not _is_valid(best_inner, final_side, outer_angle, min_sep=1e-9):
        # Fallback to seed
        best_inner = np.array(seed_inner)
        final_side = _compute_side(best_inner, outer_angle)

    # Format output
    inner_rows = []
    for row in best_inner:
        inner_rows.append([float(row[0]), float(row[1]), float(row[2])])

    center = (0.0, 0.0)
    side = float(final_side)
    angle_deg = float(outer_angle)

    return inner_rows, center, side, angle_deg
