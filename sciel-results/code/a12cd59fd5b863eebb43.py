import numpy as np
from scipy.optimize import minimize
import time

N = 11
_SQRT3 = np.sqrt(3.0)
_TOL = 1e-9


def _hex_verts(cx, cy, angle_deg):
    a = np.deg2rad(angle_deg)
    th = np.deg2rad(np.array([0, 60, 120, 180, 240, 300])) + a
    return np.column_stack([np.cos(th), np.sin(th)]) + np.array([cx, cy])


def _norm(v):
    n = np.linalg.norm(v)
    return v / n if n > _TOL else np.zeros(2)


def _min_dist_convex(P, Q):
    """Exact minimum distance between two convex polygons (>=0)."""
    # 1) Edge-edge closest points
    best = float('inf')
    for i in range(6):
        a1 = P[i]; a2 = P[(i + 1) % 6]
        d1 = a2 - a1
        for j in range(6):
            b1 = Q[j]; b2 = Q[(j + 1) % 6]
            d2 = b2 - b1
            r = a1 - b1
            a = np.dot(d1, d1); c = np.dot(d2, d2)
            b = np.dot(d1, d2); d = np.dot(d1, r); e = np.dot(d2, r)
            denom = a * c - b * b
            s = t = 0.0
            if denom <= _TOL:
                if a <= _TOL:
                    s = 0.0
                else:
                    s = np.clip(-d / a, 0.0, 1.0)
                t = np.clip((b * s + e) / c if c > _TOL else 0.0, 0.0, 1.0)
            else:
                s = np.clip((b * e - c * d) / denom, 0.0, 1.0)
                t = (b * s + e) / c
                if t < 0.0:
                    t = 0.0
                    s = np.clip(-d / a, 0.0, 1.0)
                elif t > 1.0:
                    t = 1.0
                    s = np.clip((b - d) / a, 0.0, 1.0)
            ca = a1 + s * d1
            cb = b1 + t * d2
            dd = ca - cb
            dist = np.dot(dd, dd)
            if dist < best:
                best = dist
    best = np.sqrt(best)
    # 2) Vertex-inside checks
    for V in (P, Q):
        for p in V:
            if _point_in_convex(p, Q):
                return 0.0
    for V in (Q, P):
        for p in V:
            if _point_in_convex(p, P):
                return 0.0
    return best


def _point_in_convex(p, P):
    s = 0
    for i in range(6):
        e = P[(i + 1) % 6] - P[i]
        cross = e[0] * (p[1] - P[i][1]) - e[1] * (p[0] - P[i][0])
        if cross > _TOL:
            s |= 1
        elif cross < -_TOL:
            s |= 2
    return not (s == 3)


def _containment_dist(P, outer):
    """Signed distance: positive if P inside outer hexagon, negative if outside."""
    mn = float('inf')
    for i in range(6):
        v1 = outer[i]; v2 = outer[(i + 1) % 6]
        n = _norm(np.array([- (v2[1] - v1[1]), v2[0] - v1[0]]))
        if np.dot(n, v1) > 0:
            n = -n
        for p in P:
            mn = min(mn, np.dot(p - v1, n))
    return mn


def _honeycomb_positions():
    """Compact hexagonal tiling of 11 flat-top unit hexagons, centered."""
    S = _SQRT3
    pos = []
    for r in range(3):
        for c in range(4):
            if len(pos) >= N:
                break
            x = c * 2 * S + (S if r % 2 else 0.0)
            y = r * 1.5
            pos.append([x, y])
    px = np.array([p[0] for p in pos]); py = np.array([p[1] for p in pos])
    cx = (px.max() + px.min()) / 2; cy = (py.max() + py.min()) / 2
    return np.array([[p[0] - cx, p[1] - cy] for p in pos])


def construct():
    start = time.time()
    deadline = start + 138.0
    rng = np.random.default_rng(12345)
    base = _honeycomb_positions()

    def pack(x):
        pos = x[: 3 * N].reshape(N, 2)
        ang = x[3 * N: 3 * N + N]
        outer_side = x[-2]
        outer_angle = x[-1]
        return pos, ang, outer_side, outer_angle

    def objective(x):
        return x[-2]

    def jac(x):
        g = np.zeros_like(x)
        g[-2] = 1.0
        return g

    def cons(x):
        pos, ang, outer_side, outer_angle = pack(x)
        outer = _hex_verts(0.0, 0.0, outer_angle) * outer_side
        out = []
        for i in range(N):
            out.append({'type': 'ineq',
                        'fun': (lambda x, i=i: _containment_dist(_hex_verts(x[3 * i], x[3 * i + 1], x[3 * N + i]),
                                                                 _hex_verts(0.0, 0.0, x[-1]) * x[-2])),
                        'jac': (lambda x, i=i: _jac_containment(x, i))})
        for i in range(N):
            for j in range(i + 1, N):
                out.append({'type': 'ineq',
                            'fun': (lambda x, i=i, j=j: _min_dist_convex(_hex_verts(x[3 * i], x[3 * i + 1], x[3 * N + i]),
                                                                          _hex_verts(x[3 * j], x[3 * j + 1], x[3 * N + j]))),
                            'jac': (lambda x, i=i, j=j: _jac_overlap(x, i, j))})
        return out

    def _jac_containment(x, i):
        g = np.zeros_like(x)
        pos, ang, outer_side, outer_angle = pack(x)
        P = _hex_verts(pos[i, 0], pos[i, 1], ang[i])
        outer = _hex_verts(0.0, 0.0, outer_angle) * outer_side
        best = float('inf'); best_n = None; best_v1 = None
        for k in range(6):
            v1 = outer[k]; v2 = outer[(k + 1) % 6]
            n = _norm(np.array([-(v2[1] - v1[1]), v2[0] - v1[0]]))
            if np.dot(n, v1) > 0:
                n = -n
            for p in P:
                d = np.dot(p - v1, n)
                if d < best:
                    best = d; best_n = n; best_v1 = v1
        if best_n is None:
            return g
        # d = n.(p - v1); p = C + R(ang) q
        dC = np.array([np.cos(np.deg2rad(ang[i])), np.sin(np.deg2rad(ang[i]))])
        g[3 * i] += best_n[0]
        g[3 * i + 1] += best_n[1]
        g[3 * N + i] += -np.sin(np.deg2rad(ang[i])) * best_n[0] + np.cos(np.deg2rad(ang[i])) * best_n[1]
        # outer_side: v1 = outer_side * u, d = n.(C - outer_side*u)
        u = best_v1 / outer_side if outer_side > _TOL else np.zeros(2)
        g[-2] += -np.dot(best_n, u)
        return g

    def _jac_overlap(x, i, j):
        g = np.zeros_like(x)
        pos, ang, _, _ = pack(x)
        P = _hex_verts(pos[i, 0], pos[i, 1], ang[i])
        Q = _hex_verts(pos[j, 0], pos[j, 1], ang[j])
        best = float('inf'); ca = None; cb = None
        for ii in range(6):
            a1 = P[ii]; a2 = P[(ii + 1) % 6]; d1 = a2 - a1
            for jj in range(6):
                b1 = Q[jj]; b2 = Q[(jj + 1) % 6]; d2 = b2 - b1
                r = a1 - b1
                a = np.dot(d1, d1); c = np.dot(d2, d2)
                b = np.dot(d1, d2); dd = np.dot(d1, r); e = np.dot(d2, r)
                denom = a * c - b * b
                s = t = 0.0
                if denom <= _TOL:
                    s = np.clip(-dd / a, 0.0, 1.0) if a > _TOL else 0.0
                    t = np.clip((b * s + e) / c, 0.0, 1.0) if c > _TOL else 0.0
                else:
                    s = np.clip((b * e - c * dd) / denom, 0.0, 1.0)
                    t = (b * s + e) / c
                    if t < 0.0:
                        t = 0.0; s = np.clip(-dd / a, 0.0, 1.0)
                    elif t > 1.0:
                        t = 1.0; s = np.clip((b - dd) / a, 0.0, 1.0)
                ca2 = a1 + s * d1; cb2 = b1 + t * d2
                dist = np.dot(ca2 - cb2, ca2 - cb2)
                if dist < best:
                    best = dist; ca = ca2; cb = cb2
        for p in P:
            if _point_in_convex(p, Q):
                return g
        for p in Q:
            if _point_in_convex(p, P):
                return g
        if ca is None:
            return g
        vec = ca - cb
        L = np.linalg.norm(vec)
        if L < _TOL:
            return g
        n = vec / L
        g[3 * i] += n[0]; g[3 * i + 1] += n[1]
        g[3 * j] -= n[0]; g[3 * j + 1] -= n[1]
        return g

    bounds = [(-30, 30), (-30, 30), (-180, 180)] * N + [(1.0, 60.0), (-180, 180)]

    def valid(pos, ang, outer_side, outer_angle):
        outer = _hex_verts(0.0, 0.0, outer_angle) * outer_side
        for i in range(N):
            if _containment_dist(_hex_verts(pos[i, 0], pos[i, 1], ang[i]), outer) < -1e-6:
                return False
        for i in range(N):
            for j in range(i + 1, N):
                if _min_dist_convex(_hex_verts(pos[i, 0], pos[i, 1], ang[i]),
                                    _hex_verts(pos[j, 0], pos[j, 1], ang[j])) < -1e-6:
                    return False
        return True

    def refine(x, budget):
        pos, ang, outer_side, outer_angle = pack(x)
        def f(y):
            p, a, os, oa = pack(y)
            return os
        try:
            r = minimize(f, x, method='Nelder-Mead',
                         options={'maxiter': 400, 'xatol': 1e-4, 'fatol': 1e-5,
                                  'adaptive': True, 'maxfev': 400})
            if r.x[-2] < x[-2] - 1e-7 and valid(*pack(r.x)):
                return r.x
        except Exception:
            pass
        return x

    best = None; best_side = float('inf')
    max_extent = max(np.max(np.abs(base)), 1.0) + 3.0
    x0 = np.concatenate([base.ravel(), np.zeros(N), [max_extent, 0.0]])

    for restart in range(10):
        if time.time() > deadline - 20.0:
            break
        xstart = x0.copy()
        if restart > 0:
            xstart[: 2 * N] += rng.uniform(-2.0, 2.0, size=2 * N)
            xstart[3 * N: 3 * N + N] += rng.uniform(-60, 60, size=N)
            xstart[-2] = max_extent + rng.uniform(-0.5, 2.0)
        try:
            res = minimize(objective, xstart, method='trust-constr', jac=jac,
                           bounds=bounds, constraints=cons,
                           options={'maxiter': 60, 'gtol': 1e-4, 'verbose': 0})
            cand = res.x
            if valid(*pack(cand)):
                cand = refine(cand, deadline - time.time() - 6.0)
                if cand[-2] < best_side:
                    best = cand; best_side = cand[-2]
        except Exception:
            pass
        if time.time() > deadline - 15.0:
            break

    if best is not None:
        pos, ang, outer_side, outer_angle = pack(best)
        inner = [[float(pos[i, 0]), float(pos[i, 1]), float(ang[i])] for i in range(N)]
        return inner, (0.0, 0.0), float(outer_side), float(outer_angle)
    else:
        inner = [[float(base[i, 0]), float(base[i, 1]), 0.0] for i in range(N)]
        return inner, (0.0, 0.0), float(max_extent + 4.0), 0.0
