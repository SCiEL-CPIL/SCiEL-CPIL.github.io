"""Fresh design: pack N unit hexagons inside the smallest enclosing regular hexagon.

Entry point construct() -> (inner, center, side, angle) is kept exactly as in the seed.
inner rows are [x, y, angle_degrees]; center is (0,0) (global translation is free);
side is the outer hexagon side length; angle is the outer hexagon rotation (deg).
"""
import time
import math
import random
import numpy as np
from scipy.optimize import minimize

N = 11
DEADLINE = 130.0          # wall-clock budget (evaluator kills at 150s; keep margin)
H = math.sqrt(3.0) / 2.0  # inradius of a unit regular hexagon


# ---------------- geometry: unit hexagon ----------------
def unit_hex_vertices(angle_deg):
    a = math.radians(angle_deg)
    cs, sn = math.cos(a), math.sin(a)
    v = np.empty((6, 2))
    for k in range(6):
        th = math.radians(60.0 * k)
        x, y = math.cos(th), math.sin(th)
        v[k, 0] = x * cs - y * sn
        v[k, 1] = x * sn + y * cs
    return v


def hex_outward_normals(angle_deg):
    a = math.radians(angle_deg)
    cs, sn = math.cos(a), math.sin(a)
    n = np.empty((6, 2))
    for k in range(6):
        th = math.radians(60.0 * k + 30.0)
        x, y = math.cos(th), math.sin(th)
        n[k, 0] = x * cs - y * sn
        n[k, 1] = x * sn + y * cs
    return n


def _seg_seg_dist(p1, p2, q1, q2):
    d1 = p2 - p1
    d2 = q2 - q1
    r = p1 - q1
    a = float(d1 @ d1)
    e = float(d2 @ d2)
    f = float(d2 @ r)
    if a <= 1e-12 and e <= 1e-12:
        return float(np.linalg.norm(r))
    if a <= 1e-12:
        s = min(max(f, 0.0), e)
        d = r + s * d2
    else:
        c = float(d1 @ r)
        if e <= 1e-12:
            t = min(max(-c, 0.0), a)
            d = r - t * d1
        else:
            b = float(d1 @ d2)
            den = a * e - b * b
            t = (b * f - c * e) / den if den > 1e-12 else 0.0
            if t < 0.0:
                t = 0.0
                s = min(max(f, 0.0), e)
            elif t > 1.0:
                t = 1.0
                s = min(max(f + b, 0.0), e)
            else:
                s = (b * t + f) / e if e > 1e-12 else 0.0
                if s < 0.0:
                    s = 0.0
                elif s > 1.0:
                    s = 1.0
                    t = min(max(-c + b * s, 0.0), a)
            d = r - t * d1 + s * d2
    return float(np.linalg.norm(d))


def min_sep(v1, v2):
    best = float('inf')
    n1, n2 = v1.shape[0], v2.shape[0]
    for i in range(n1):
        a = v1[i]
        b = v1[(i + 1) % n1]
        for j in range(n2):
            c = v2[j]
            d = v2[(j + 1) % n2]
            dval = _seg_seg_dist(a, b, c, d)
            if dval < best:
                best = dval
                if best < 1e-9:
                    return 0.0
    return best


# ---------------- outer hexagon ----------------
def outer_side_for(vertices, phi_deg):
    n = hex_outward_normals(phi_deg)
    d = vertices @ n.T
    return float(np.max(d) / H)


def outer_side_min(vertices, phi_grid):
    best = float('inf')
    best_phi = 0.0
    for phi in phi_grid:
        s = outer_side_for(vertices, phi)
        if s < best:
            best = s
            best_phi = phi
    return best, best_phi


def enclosing_radius_max(vertices):
    return float(np.max(np.linalg.norm(vertices, axis=1)))


def _all_vertices(pos, ang):
    n = pos.shape[0]
    verts = np.empty((n * 6, 2))
    for i in range(n):
        verts[6 * i: 6 * i + 6] = pos[i] + unit_hex_vertices(ang[i])
    return verts


def overlap_penalty(pos, ang):
    n = pos.shape[0]
    verts = [None] * n
    for i in range(n):
        verts[i] = pos[i] + unit_hex_vertices(ang[i])
    pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = min_sep(verts[i], verts[j])
            if d < 0.0:
                p = -d
                pen += p * p
    return pen


def objective(x):
    pos = x[:2 * N].reshape(N, 2)
    ang = x[2 * N:]
    verts = _all_vertices(pos, ang)
    phi_grid = np.arange(0.0, 180.0, 15.0)
    s, _ = outer_side_min(verts, phi_grid)
    return s + 50.0 * overlap_penalty(pos, ang)


def _refine_side(pos, ang, t_end):
    verts = _all_vertices(pos, ang)
    R = enclosing_radius_max(verts)
    lo, hi = 0.0, R + 1e-6
    for _ in range(34):
        if time.time() > t_end:
            break
        mid = 0.5 * (lo + hi)
        ok = False
        for phi in np.arange(0.0, 180.0, 3.0):
            if outer_side_for(verts, phi) <= mid + 1e-9:
                ok = True
                break
        if ok:
            hi = mid
        else:
            lo = mid
    s = hi
    best_phi = 0.0
    best_s = float('inf')
    for phi in np.arange(0.0, 180.0, 0.5):
        sv = outer_side_for(verts, phi)
        if sv <= best_s:
            best_s = sv
            best_phi = phi
    return s, best_phi


def construct():
    t0 = time.time()
    t_end = t0 + DEADLINE
    rng = random.Random(12345)
    np_rng = np.random.default_rng(12345)

    def lattice_seed():
        pos = np.empty((N, 2))
        ang = np.zeros(N)
        a = 2.0
        b = math.sqrt(3.0)
        idx = 0
        for i in range(-2, 3):
            for j in range(-2, 3):
                if idx >= N:
                    break
                pos[idx, 0] = i * a + j * (a / 2.0)
                pos[idx, 1] = j * b
                idx += 1
        pos -= pos.mean(axis=0)
        return pos, ang

    best_pos, best_ang = lattice_seed()
    best_score = float('inf')

    def consider(pos, ang, score):
        nonlocal best_pos, best_ang, best_score
        if score < best_score:
            best_score = score
            best_pos = pos.copy()
            best_ang = ang.copy()

    x0 = np.concatenate([best_pos.ravel(), best_ang])
    try:
        res = minimize(objective, x0, method='Nelder-Mead',
                       options={'maxiter': 1200, 'xatol': 1e-5, 'fatol': 1e-6, 'adaptive': True})
        if res.fun < best_score:
            consider(res.x[:2 * N].reshape(N, 2), res.x[2 * N:], res.fun)
    except Exception:
        pass
    if time.time() > t_end - 20.0:
        s, phi = _refine_side(best_pos, best_ang, t_end)
        inner = [[float(best_pos[i, 0]), float(best_pos[i, 1]), float(best_ang[i])] for i in range(N)]
        return inner, (0.0, 0.0), float(s), float(phi)

    while time.time() < t_end - 30.0:
        if rng.random() < 0.7:
            pos, ang = lattice_seed()
            pos += np_rng.normal(0.0, 0.6, pos.shape)
            ang = np_rng.uniform(-30.0, 30.0, N)
        else:
            pos = np_rng.uniform(-6.0, 6.0, (N, 2))
            ang = np_rng.uniform(-30.0, 30.0, N)
        x0 = np.concatenate([pos.ravel(), ang])
        try:
            s0 = objective(x0)
        except Exception:
            continue
        if s0 >= best_score and s0 > best_score + 2.0:
            continue
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                           options={'maxiter': 900, 'xatol': 1e-5, 'fatol': 1e-6, 'adaptive': True})
        except Exception:
            continue
        if res.fun < best_score:
            consider(res.x[:2 * N].reshape(N, 2), res.x[2 * N:], res.fun)

    s, phi = _refine_side(best_pos, best_ang, t_end)
    inner = [[float(best_pos[i, 0]), float(best_pos[i, 1]), float(best_ang[i])] for i in range(N)]
    return inner, (0.0, 0.0), float(s), float(phi)
