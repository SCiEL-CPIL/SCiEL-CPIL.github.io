import math
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 11
D = math.sqrt(3)
SQ3 = math.sqrt(3)

def _hex_vertices(cx, cy, angle_deg, s=1.0):
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([s * np.cos(angles), s * np.sin(angles)])
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
        t = np.clip(f / e, 0, 1)
        return np.linalg.norm(p1 - (p3 + t * d2))
    c = np.dot(d1, r)
    if e < 1e-12:
        s = np.clip(c / a, 0, 1)
        return np.linalg.norm((p1 + s * d1) - p3)
    b = np.dot(d1, d2)
    denom = a * e - b * b
    s = 0.0
    t = 0.0
    if denom > 1e-12:
        s = (b * f - c * e) / denom
        s = np.clip(s, 0, 1)
        t = (b * s + f) / e
        if t < 0:
            t = 0.0
            s = np.clip(c / a, 0, 1)
        elif t > 1:
            t = 1.0
            s = np.clip((b + f) / a, 0, 1)
        return np.linalg.norm((p1 + s * d1) - (p3 + t * d2))
    else:
        t = np.clip(f / e, 0, 1)
        return np.linalg.norm(p1 - (p3 + t * d2))

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

def _outer_hex_vertices(cx, cy, side, angle_deg):
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([side * np.cos(angles), side * np.sin(angles)])
    return vs + np.array([cx, cy])

_BASE = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]))
_BASE_COS = np.cos(_BASE)
_BASE_SIN = np.sin(_BASE)
_APO = SQ3 / 2.0

def _compute_side(centers, angles_rad, outer_angle_rad):
    """Exact min outer side length. centers (n,2), angles_rad (n,), outer_angle_rad scalar."""
    na = outer_angle_rad + np.deg2rad(30 + np.arange(6) * 60)
    normals = np.column_stack([np.cos(na), np.sin(na)])  # (6,2)
    n = centers.shape[0]
    ca = np.cos(angles_rad)
    sa = np.sin(angles_rad)
    # rotated unit vertex offsets (n,6,2)
    ox = _BASE_COS * ca[:, None] - _BASE_SIN * sa[:, None]
    oy = _BASE_COS * sa[:, None] + _BASE_SIN * ca[:, None]
    vx = centers[:, 0:1] + ox
    vy = centers[:, 1:2] + oy
    proj = vx @ normals[0] + vy @ normals[1]  # (n,6)
    return float(np.max(np.abs(proj))) / _APO

def _smooth_side(centers, angles_rad, outer_angle_rad, beta=80.0):
    """Smooth (log-sum-exp) surrogate of max |projection|, and its gradient w.r.t. centers."""
    na = outer_angle_rad + np.deg2rad(30 + np.arange(6) * 60)
    normals = np.column_stack([np.cos(na), np.sin(na)])  # (6,2)
    n = centers.shape[0]
    ca = np.cos(angles_rad)
    sa = np.sin(angles_rad)
    ox = _BASE_COS * ca[:, None] - _BASE_SIN * sa[:, None]
    oy = _BASE_COS * sa[:, None] + _BASE_SIN * ca[:, None]
    vx = centers[:, 0:1] + ox
    vy = centers[:, 1:2] + oy
    proj = vx @ normals[0] + vy @ normals[1]  # (n,6)
    # log-sum-exp over (n*6) of proj^2
    sq = proj * proj
    m = np.max(sq)
    Z = m + np.log(np.sum(np.exp(sq - m)))
    w = np.exp(sq - Z)  # (n,6) weights summing to 1
    # d/dproj of Z = 2*proj*w ; dproj/dcenter_i = normals[k]
    grad = 2.0 * (w @ (proj))  # (n,)
    gx = grad @ normals[0]
    gy = grad @ normals[1]
    val = 0.5 * Z / _APO
    g = np.column_stack([gx, gy]) / _APO
    return val, g

def _compute_side_inner(inner_params, outer_angle_deg):
    centers = inner_params[:, :2]
    angles_rad = np.deg2rad(inner_params[:, 2])
    return _compute_side(centers, angles_rad, np.deg2rad(outer_angle_deg))

def _overlap_penalty(inner_params, penalty_weight=50.0, thresh=0.005):
    n = len(inner_params)
    penalty = 0.0
    for i in range(n):
        v1 = _hex_vertices(inner_params[i, 0], inner_params[i, 1], inner_params[i, 2])
        for j in range(i+1, n):
            v2 = _hex_vertices(inner_params[j, 0], inner_params[j, 1], inner_params[j, 2])
            sep = _poly_separation(v1, v2)
            if sep < thresh:
                penalty += penalty_weight * (thresh - sep) ** 2
    return penalty

def _is_valid(inner_params, side, outer_angle_deg, min_sep=0.0):
    n = len(inner_params)
    outer_v = _outer_hex_vertices(0.0, 0.0, side, outer_angle_deg)
    for i in range(n):
        hex_v = _hex_vertices(inner_params[i, 0], inner_params[i, 1], inner_params[i, 2])
        for v in hex_v:
            if not _point_in_polygon(v, outer_v):
                return False
    for i in range(n):
        v1 = _hex_vertices(inner_params[i, 0], inner_params[i, 1], inner_params[i, 2])
        for j in range(i+1, n):
            v2 = _hex_vertices(inner_params[j, 0], inner_params[j, 1], inner_params[j, 2])
            if _poly_separation(v1, v2) < min_sep:
                return False
    return True

def _seed_solution():
    eps = 1e-9
    centers = [(0.0, 0.0)]
    for k in range(6):
        a = np.deg2rad(k * 60)
        centers.append((SQ3 + eps) * np.cos(a), (SQ3 + eps) * np.sin(a))
    for k in range(4):
        a = np.deg2rad(30 + k * 60)
        centers.append((3.0 + 2*eps) * np.cos(a), (3.0 + 2*eps) * np.sin(a))
    inner = [[x, y, 30.0] for (x, y) in centers]
    arr = np.array(inner)
    side = _compute_side_inner(arr, 30.0)
    return inner, (0.0, 0.0), side, 30.0

def _lattice_packing():
    v1 = np.array([SQ3, 0.0])
    v2 = np.array([SQ3/2, 1.5])
    selected = [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1),(1,1),(-1,-1),(2,0),(0,-2)]
    centers = [i*v1 + j*v2 for (i, j) in selected]
    inner = [[float(p[0]), float(p[1]), 30.0] for p in centers]
    arr = np.array(inner)
    best_side = float('inf')
    best_angle = 0.0
    for a in np.arange(0, 60, 0.5):
        s = _compute_side_inner(arr, a)
        if s < best_side:
            best_side, best_angle = s, a
    for a in np.arange(best_angle - 1, best_angle + 1, 0.05):
        s = _compute_side_inner(arr, a)
        if s < best_side:
            best_side, best_angle = s, a
    return inner, (0.0, 0.0), best_side, best_angle

def _polish_angle(inner_arr, initial_angle, time_limit):
    best_side = _compute_side_inner(inner_arr, initial_angle)
    best_angle = initial_angle
    for a in np.arange(initial_angle - 5, initial_angle + 5, 0.5):
        if time.time() > time_limit:
            break
        s = _compute_side_inner(inner_arr, a)
        if s < best_side:
            best_side, best_angle = s, a
    for a in np.arange(best_angle - 0.5, best_angle + 0.5, 0.05):
        if time.time() > time_limit:
            break
        s = _compute_side_inner(inner_arr, a)
        if s < best_side:
            best_side, best_angle = s, a
    if time.time() < time_limit:
        try:
            res = minimize_scalar(lambda a: _compute_side_inner(inner_arr, a),
                                  bounds=(best_angle - 0.1, best_angle + 0.1),
                                  method='bounded', options={'xatol': 1e-12})
            if res.success and res.fun < best_side:
                best_side, best_angle = res.fun, res.x
        except Exception:
            pass
    if time.time() < time_limit:
        try:
            res2 = minimize_scalar(lambda a: _compute_side_inner(inner_arr, a),
                                   bounds=(best_angle - 0.01, best_angle + 0.01),
                                   method='bounded', options={'xatol': 1e-14})
            if res2.success and res2.fun < best_side:
                best_side, best_angle = res2.fun, res2.x
        except Exception:
            pass
    return best_side, best_angle

def _objective(params, outer_angle_deg, n):
    inner_params = params.reshape(n, 3)
    side = _compute_side_inner(inner_params, outer_angle_deg)
    return side + _overlap_penalty(inner_params)

def _gradient_polish(inner_arr, outer_angle_deg, time_limit):
    """Smooth-surrogate gradient descent on the 22 position coordinates."""
    n = inner_arr.shape[0]
    centers = inner_arr[:, :2].copy()
    angles_rad = np.deg2rad(inner_arr[:, 2].copy())
    oar = np.deg2rad(outer_angle_deg)
    best_val = _compute_side(centers, angles_rad, oar)
    best_centers = centers.copy()

    def obj_g(c):
        return _smooth_side(c, angles_rad, oar, beta=80.0)

    # L-BFGS-B on positions
    if time.time() < time_limit:
        try:
            x0 = centers.reshape(-1)
            def f(x):
                return _smooth_side(x.reshape(n, 2), angles_rad, oar, beta=80.0)[0]
            def g(x):
                return _smooth_side(x.reshape(n, 2), angles_rad, oar, beta=80.0)[1].reshape(-1)
            res = minimize(f, x0, jac=g, method='L-BFGS-B',
                           options={'maxiter': 400, 'gtol': 1e-10, 'ftol': 1e-12})
            c2 = res.x.reshape(n, 2)
            v2 = _compute_side(c2, angles_rad, oar)
            if v2 < best_val:
                best_val, best_centers = v2, c2
        except Exception:
            pass

    # Refine inner angles with smooth surrogate (per-hexagon 1D)
    if time.time() < time_limit:
        for i in range(n):
            def fa(a):
                ang = angles_rad.copy()
                ang[i] = np.deg2rad(a)
                return _smooth_side(best_centers, ang, oar, beta=80.0)[0]
            try:
                r = minimize_scalar(fa, bounds=(float(np.rad2deg(angles_rad[i])) - 30,
                                                float(np.rad2deg(angles_rad[i])) + 30),
                                    method='bounded', options={'xatol': 1e-8})
                ang = angles_rad.copy()
                ang[i] = np.deg2rad(r.x)
                v = _compute_side(best_centers, ang, oar)
                if v < best_val:
                    best_val, angles_rad = v, ang
            except Exception:
                pass

    out = np.column_stack([best_centers, np.rad2deg(angles_rad)])
    return out, best_val

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    safety_margin = 5.0
    n = 11

    seed_candidates = []
    s1_inner, _, s1_side, s1_angle = _seed_solution()
    seed_candidates.append((s1_inner, s1_side, s1_angle))
    s2_inner, _, s2_side, s2_angle = _lattice_packing()
    seed_candidates.append((s2_inner, s2_side, s2_angle))

    best_seed = min(seed_candidates, key=lambda x: x[1])
    best_inner = [list(r) for r in best_seed[0]]
    best_side = best_seed[1]
    best_angle = best_seed[2]

    # Phase 1: Nelder-Mead over positions for several outer angles
    init_params = np.zeros(3 * n)
    for i, (x, y, a) in enumerate(best_seed[0]):
        init_params[i*3], init_params[i*3+1], init_params[i*3+2] = x, y, a

    for outer_angle in [best_angle, 0.0, 15.0, 30.0, 45.0]:
        if time.time() > DEADLINE - safety_margin - 25:
            break
        try:
            res = minimize(_objective, init_params.copy(), args=(outer_angle, n),
                           method='Nelder-Mead',
                           options={'maxiter': 1500, 'xatol': 1e-7, 'fatol': 1e-8, 'adaptive': True})
            if time.time() > DEADLINE - safety_margin:
                break
            cand = res.x.reshape(n, 3)
            cs = _compute_side_inner(cand, outer_angle)
            if _is_valid(cand, cs, outer_angle, min_sep=1e-8) and cs < best_side:
                best_inner, best_side, best_angle = cand.tolist(), cs, outer_angle
        except Exception:
            pass

    # Phase 2: angle polish
    if time.time() < DEADLINE - safety_margin:
        best_side, best_angle = _polish_angle(np.array(best_inner), best_angle, DEADLINE - safety_margin)

    # Phase 3: gradient polish (smooth surrogate)
    if time.time() < DEADLINE - safety_margin - 10:
        try:
            g_inner, g_side = _gradient_polish(np.array(best_inner), best_angle, DEADLINE - safety_margin)
            if _is_valid(g_inner, g_side, best_angle, min_sep=1e-8) and g_side < best_side:
                best_inner, best_side = g_inner.tolist(), g_side
        except Exception:
            pass

    # Phase 4: refine positions with exact objective
    if time.time() < DEADLINE - safety_margin - 10:
        try:
            init2 = np.array(best_inner).reshape(-1)
            res2 = minimize(_objective, init2, args=(best_angle, n),
                            method='Nelder-Mead',
                            options={'maxiter': 800, 'xatol': 1e-9, 'fatol': 1e-10, 'adaptive': True})
            if time.time() < DEADLINE - safety_margin:
                cand2 = res2.x.reshape(n, 3)
                cs2 = _compute_side_inner(cand2, best_angle)
                if _is_valid(cand2, cs2, best_angle, min_sep=1e-8) and cs2 < best_side:
                    best_inner, best_side = cand2.tolist(), cs2
        except Exception:
            pass

    # Phase 5: final angle polish
    if time.time() < DEADLINE - safety_margin:
        best_side, best_angle = _polish_angle(np.array(best_inner), best_angle, DEADLINE - safety_margin)

    # Final validation
    inner_arr_final = np.array(best_inner)
    final_side = _compute_side_inner(inner_arr_final, best_angle)
    if not _is_valid(inner_arr_final, final_side, best_angle, min_sep=1e-8):
        best_inner = [list(r) for r in best_seed[0]]
        best_side, best_angle = best_seed[1], best_seed[2]
        final_side = best_side
    else:
        best_side = final_side

    inner_rows = [[float(r[0]), float(r[1]), float(r[2])] for r in best_inner]
    return inner_rows, (0.0, 0.0), float(best_side), float(best_angle)
