"""Packing 11 unit regular hexagons inside a regular hexagon.

Soft-body SDF model:
  * each inner hexagon is a rotated regular hexagon with side 1 (circumradius 1)
  * a proper signed distance field for a hexagon = min over 6 half-plane distances,
    smoothed with a log-sum-exp so it is differentiable everywhere
  * overlap penalty  = (soft SDF of B evaluated at A's centroid, clamped to <= 0)^2
  * boundary penalty = (soft SDF of the outer hexagon evaluated at each inner centroid)^2
  * outer side length is a free variable
Optimized with L-BFGS-B using analytic gradients, plus simulated-annealing restarts.
"""
import numpy as np
import time
import random
import math
from scipy.optimize import minimize

N = 11
DEADLINE = 150.0
START_TIME = time.time()
SQRT3 = math.sqrt(3.0)
INR = SQRT3 / 2.0          # inradius of a unit hexagon

# Precomputed local geometry (unit hexagon, circumradius 1)
_VERT_A = np.array([k * math.pi / 3.0 for k in range(6)])
_VERTS = np.column_stack([np.cos(_VERT_A), np.sin(_VERT_A)])          # (6,2)
_NORM_A = np.array([math.radians(30.0 + 60.0 * k) for k in range(6)])
_NORMALS = np.column_stack([np.cos(_NORM_A), np.sin(_NORM_A)])        # (6,2)
_EDGE_V = (_VERTS - np.roll(_VERTS, 1, axis=0))                        # (6,2)
_EDGE_L = np.linalg.norm(_EDGE_V, axis=1)                              # (6,)
_EDGE_N = np.column_stack([-_EDGE_V[:, 1] / _EDGE_L, _EDGE_V[:, 0] / _EDGE_L])  # (6,2)


def _remaining_time():
    return DEADLINE - (time.time() - START_TIME)


def _rot_mat(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def _softmin(d, beta):
    """Smooth minimum of 1-D array d (log-sum-exp of -d). Returns (value, grad w.r.t. d)."""
    m = np.min(d)
    e = np.exp((m - d) / beta)
    z = np.sum(e)
    val = m - beta * math.log(z)
    w = e / z
    return val, w


def _hex_sdf(px, py, cx, cy, theta, side, beta):
    """Smooth signed distance from (px,py) to hexagon (center, rotation, side)."""
    R = _rot_mat(-theta)
    p = R @ np.array([px - cx, py - cy])
    d = p @ _NORMALS.T - side * INR
    return _softmin(d, beta)


def _overlap_penalty(a, b, beta):
    """a,b = (cx,cy,theta). Returns (penalty, grad_a(3,), grad_b(3,))."""
    cx1, cy1, th1 = a
    cx2, cy2, th2 = b
    R = _rot_mat(-th2)
    p = R @ np.array([cx1 - cx2, cy1 - cy2])
    d = p @ _NORMALS.T - INR
    sm, w = _softmin(d, beta)
    if sm >= 0.0:
        return 0.0, np.zeros(3), np.zeros(3)
    pen = sm * sm
    # d(sm)/d(p) = -w
    gd_p = -w
    gd_c1 = R.T @ gd_p
    gd_c2 = -R.T @ gd_p
    gd_t1 = np.array([np.sum(gd_p * np.array([p[1], -p[0]]))])
    gd_t2 = np.array([np.sum(gd_p * np.array([-p[1], p[0]]))])
    ga = np.array([gd_c1[0], gd_c1[1], gd_t1[0]])
    gb = np.array([gd_c2[0], gd_c2[1], gd_t2[0]])
    return pen, ga, gb


def _boundary_pen(i, S, oa, beta):
    """i=(cx,cy,theta). Returns (penalty, grad(3,))."""
    cx, cy, th = i
    c, s = math.cos(th), math.sin(th)
    pen = 0.0
    gx = gy = gt = 0.0
    for k in range(6):
        vx = cx + _VERTS[k, 0] * c - _VERTS[k, 1] * s
        vy = cy + _VERTS[k, 0] * s + _VERTS[k, 1] * c
        val, w = _hex_sdf(vx, vy, 0.0, 0.0, oa, S, beta)
        if val >= 0.0:
            continue
        pen += val * val
        # d(val)/d(v) = -w ;  d(v)/d(c) = [vx,vy] ; d(v)/d(s) = [-vy,vx] ; d(v)/d(t) = [-vy,vx]
        dvx, dvy = -w[0], -w[1]
        gx += dvx * vx
        gy += dvx * vy
        gt += dvx * (-vy) + dvy * vx
    return pen, np.array([gx, gy, gt])


def _objective(x, oa, beta):
    pos = x[:3 * N].reshape(N, 3)
    S = x[3 * N]
    f = S
    for i in range(N):
        p, _ = _boundary_pen(pos[i], S, oa, beta)
        f += p
    for i in range(N):
        for j in range(i + 1, N):
            p, _, _ = _overlap_penalty(pos[i], pos[j], beta)
            f += p
    return f


def _gradient(x, oa, beta):
    pos = x[:3 * N].reshape(N, 3)
    S = x[3 * N]
    g = np.zeros_like(x)
    g[3 * N] = 1.0  # d(S)/dS
    for i in range(N):
        _, gi = _boundary_pen(pos[i], S, oa, beta)
        g[3 * i:3 * i + 3] += gi
    for i in range(N):
        for j in range(i + 1, N):
            _, ga, gb = _overlap_penalty(pos[i], pos[j], beta)
            g[3 * i:3 * i + 3] += ga
            g[3 * j:3 * j + 3] += gb
    return g


def _min_enclosing_side(positions, oa):
    c, s = math.cos(-oa), math.sin(-oa)
    best = 0.0
    for (ix, iy, iangle) in positions:
        th = math.radians(iangle)
        ci, si = math.cos(th), math.sin(th)
        for k in range(6):
            vx = ix + _VERTS[k, 0] * ci - _VERTS[k, 1] * si
            vy = iy + _VERTS[k, 0] * si + _VERTS[k, 1] * ci
            dx, dy = vx, vy
            lx = dx * c - dy * s
            ly = dx * s + dy * c
            proj = lx * _NORMALS[:, 0] + ly * _NORMALS[:, 1]
            if proj[0] > 0:
                m = np.max(proj)
                if m > best:
                    best = m
    return best * 2.0 / SQRT3


def _hexagons_disjoint(a, b, tol=1e-7):
    def verts(cx, cy, th):
        c, s = math.cos(th), math.sin(th)
        return np.column_stack([cx + _VERTS[:, 0] * c - _VERTS[:, 1] * s,
                                cy + _VERTS[:, 0] * s + _VERTS[:, 1] * c])
    v1 = verts(a[0], a[1], math.radians(a[2]))
    v2 = verts(b[0], b[1], math.radians(b[2]))
    e1 = v1 - np.roll(v1, 1, axis=0)
    e2 = v2 - np.roll(v2, 1, axis=0)
    for e in (e1, e2):
        ln = np.linalg.norm(e, axis=1)
        n = np.column_stack([-e[:, 1] / ln, e[:, 0] / ln])
        p1 = v1 @ n.T
        p2 = v2 @ n.T
        if np.max(p1) < np.min(p2) - tol or np.max(p2) < np.min(p1) - tol:
            return True
    return False


def _check_validity(positions, oa, tol=1e-4):
    S = _min_enclosing_side(positions, oa)
    c, s = math.cos(-oa), math.sin(-oa)
    lim = S * INR + tol
    for (ix, iy, iangle) in positions:
        th = math.radians(iangle)
        ci, si = math.cos(th), math.sin(th)
        for k in range(6):
            vx = ix + _VERTS[k, 0] * ci - _VERTS[k, 1] * si
            vy = iy + _VERTS[k, 0] * si + _VERTS[k, 1] * ci
            lx = vx * c - vy * s
            ly = vx * s + vy * c
            proj = lx * _NORMALS[:, 0] + ly * _NORMALS[:, 1]
            if np.max(proj) > lim:
                return False
    for i in range(N):
        for j in range(i + 1, N):
            if not _hexagons_disjoint(positions[i], positions[j], tol=tol):
                return False
    return True


def _lattice():
    pos = [(0.0, 0.0, 0.0)]
    d1 = 2.0 * SQRT3
    for k in range(6):
        a = math.radians(60.0 * k)
        pos.append((d1 * math.cos(a), d1 * math.sin(a), 0.0))
    d2 = 6.0
    for k in range(4):
        a = math.radians(30.0 + 60.0 * k)
        pos.append((d2 * math.cos(a), d2 * math.sin(a), 0.0))
    return pos


def _ring():
    pos = [(0.0, 0.0, 0.0)]
    for i in range(6):
        a = 2 * math.pi * i / 6
        pos.append((3.4 * math.cos(a), 3.4 * math.sin(a), 0.0))
    for i in range(4):
        a = 2 * math.pi * (i + 0.5) / 4
        pos.append((5.5 * math.cos(a), 5.5 * math.sin(a), 0.0))
    return pos


def _random_cfg(seed):
    rng = random.Random(seed)
    pos = [(0.0, 0.0, 0.0)]
    for i in range(1, N):
        r = rng.uniform(2.0, 5.5)
        a = rng.uniform(0, 2 * math.pi)
        pos.append((r * math.cos(a), r * math.sin(a), rng.uniform(0, 60.0)))
    return pos


def _optimize(x0, oa, beta, budget):
    t0 = time.time()
    try:
        res = minimize(_objective, x0, jac=_gradient, args=(oa, beta),
                       method='L-BFGS-B',
                       options={'maxiter': 600, 'maxfun': 30000,
                                'ftol': 1e-15, 'gtol': 1e-9, 'maxls': 40})
    except Exception:
        return x0
    if time.time() - t0 > budget:
        return x0
    return res.x


def _anneal(oa, best_score, budget):
    t0 = time.time()
    cfgs = [_lattice(), _ring(), _random_cfg(1), _random_cfg(2), _random_cfg(3)]
    best_x = None
    best_side = float('inf')
    for init in cfgs:
        if time.time() - t0 > budget or _remaining_time() < 15.0:
            break
        x0 = np.array([p for pos in init for p in pos] + [_min_enclosing_side(init, oa)],
                      dtype=np.float64)
        for beta in (0.15, 0.05, 0.02):
            if time.time() - t0 > budget or _remaining_time() < 15.0:
                break
            x = _optimize(x0, oa, beta, 10.0)
            pos = [tuple(p) for p in x[:3 * N].reshape(N, 3)]
            side = _min_enclosing_side(pos, oa)
            if side < best_side:
                best_side = side
                best_x = x
        x0 = best_x if best_x is not None else x0
    return best_x, best_side


def construct():
    random.seed(42)
    np.random.seed(42)

    best_result = None
    best_score = float('inf')

    for oa in (0.0, 30.0):
        if _remaining_time() < 30.0:
            break
        x, side = _anneal(oa, best_score, 55.0)
        if x is None:
            continue
        pos = [tuple(p) for p in x[:3 * N].reshape(N, 3)]
        # final polish with a very smooth beta
        x2 = _optimize(x, oa, 0.01, 12.0)
        pos2 = [tuple(p) for p in x2[:3 * N].reshape(N, 3)]
        side2 = _min_enclosing_side(pos2, oa)
        if side2 < side:
            pos, side = pos2, side2
        if _check_validity(pos, oa, tol=1e-4) and side < best_score:
            best_score = side
            best_result = (pos, (0.0, 0.0), side, oa)

    if best_result is not None:
        return best_result

    pos = _lattice()
    side = _min_enclosing_side(pos, 0.0)
    return pos, (0.0, 0.0), side, 0.0
