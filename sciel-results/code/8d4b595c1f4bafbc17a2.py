# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_DEADLINE = time.time() + 620.0   # explicit guard; return best before 630s wall-clock
_EPS = 1e-6

def _hex_verts(cx, cy, s, a):
    th = np.deg2rad(a) + np.deg2rad(60.0) * np.arange(6)
    return np.column_stack((cx + s * np.cos(th), cy + s * np.sin(th)))

def _outer_side(inner, tol=1e-9):
    """Smallest outer side S (centered at origin) so every inner vertex is inside
    the outer hexagon of side S (flat-top, angle 0)."""
    X = inner[:, 0]
    Y = inner[:, 1]
    A = np.deg2rad(inner[:, 2])
    V = np.column_stack((X + np.cos(A), Y + np.sin(A)))
    # outer flat-top edges (outward normals)
    N = np.array([[math.cos(t), math.sin(t)] for t in
                  (90, 150, 210, 270, 330, 30)])
    D = V @ N.T
    M = D.max(axis=0)
    S = np.maximum(M / N[:, 1], M / N[:, 0])
    S = np.maximum(S, 0.0)
    S = np.max(S)
    if not np.isfinite(S):
        return np.inf
    return S + tol

def _valid(inner, outer, S):
    """Return (valid, outer_side). outer=(x,y,a)."""
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)):
        return False, np.inf
    if S < math.sqrt(11.0) - 1e-12:
        return False, np.inf
    ox, oy, oa = outer
    # shifted inner = inner - (ox,oy); shift preserves edge lengths -> unit check
    if np.any(np.abs(inner[:, 0] - ox) > 1e30) or np.any(np.abs(inner[:, 1] - oy) > 1e30):
        return False, np.inf
    # unit edge length (shift invariant, but verify)
    th = np.deg2rad(0.0) + np.deg2rad(60.0) * np.arange(6)
    e0 = np.array([np.cos(th), np.sin(th)])
    e1 = np.array([np.cos(th + np.deg2rad(60.0)), np.sin(th + np.deg2rad(60.0))])
    for i in range(inner.shape[0]):
        p = _hex_verts(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
        p = p - np.array([ox, oy])
        d = np.diff(p, axis=0, append=p[:1])
        for k in range(6):
            v = d[k]
            if abs(np.linalg.norm(v) - 1.0) > 1e-8:
                return False, np.inf
    # containment: every inner vertex p (shifted) inside outer (shifted to origin)
    P = np.column_stack((inner[:, 0] - ox, inner[:, 1] - oy))
    A = np.deg2rad(inner[:, 2])
    V = P + np.column_stack((np.cos(A), np.sin(A)))          # 66 vertices
    ov = _hex_verts(0.0, 0.0, S, oa)                         # outer at origin
    n_out = ov.shape[0]
    for e in range(n_out):
        p1 = ov[e]
        p2 = ov[(e + 1) % n_out]
        evec = p2 - p1
        cross = (V[:, 0] - p1[0]) * evec[1] - (V[:, 1] - p1[1]) * evec[0]
        if np.any(cross < -1e-6):
            return False, np.inf
    # pairwise separation on 12 edge normals
    m = inner.shape[0]
    for i in range(m):
        pi = _hex_verts(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
        ni = np.zeros((6, 2))
        for e in range(6):
            a = pi[e]
            b = pi[(e + 1) % 6]
            t = b - a
            nl = np.linalg.norm(t)
            ni[e] = np.array([-t[1], t[0]]) / nl
        for j in range(i + 1, m):
            pj = _hex_verts(inner[j, 0], inner[j, 1], 1.0, inner[j, 2])
            sep = False
            for e in range(6):
                n = ni[e]
                di = pi @ n
                dj = pj @ n
                if di.min() > dj.max() + _EPS or dj.min() > di.max() + _EPS:
                    sep = True
                    break
            if sep:
                continue
            sep = False
            for e in range(6):
                a = pj[e]
                b = pj[(e + 1) % 6]
                t = b - a
                nl = np.linalg.norm(t)
                n = np.array([-t[1], t[0]]) / nl
                di = pi @ n
                dj = pj @ n
                if di.min() > dj.max() + _EPS or dj.min() > di.max() + _EPS:
                    sep = True
                    break
            if not sep:
                return False, np.inf
    return True, S

def _nm(x0, f, max_iter=140, step=1.5, tol=1e-8):
    n = x0.size
    x0 = np.asarray(x0, float).copy()
    xk = [x0.copy()]
    for k in range(1, n + 1):
        d = x0.copy(); d[k - 1] += step
        xk.append(d)
    fx = [f(x) for x in xk]
    best = 0
    for it in range(max_iter):
        if it % 20 == 0 and time.time() > _DEADLINE - 10.0:
            break
        order = np.argsort(fx)
        xk = [xk[i] for i in order]; fx = [fx[i] for i in order]
        best = 0
        xbar = np.mean(xk[:-1], axis=0)
        xr = xbar + (xbar - xk[-1])
        fr = f(xr)
        if fr < fx[0]:
            xe = xbar + 2.0 * (xbar - xk[-1]); fe = f(xe)
            if fe < fr:
                xk[-1] = xe; fx[-1] = fe
            else:
                xk[-1] = xr; fx[-1] = fr
        elif fr < fx[-2]:
            xk[-1] = xr; fx[-1] = fr
        else:
            xc = xbar + 0.5 * (xk[-1] - xbar); fc = f(xc)
            if fc < fx[-1]:
                xk[-1] = xc; fx[-1] = fc
            else:
                for i in range(1, n + 1):
                    xk[i] = 0.5 * (xk[i] + xk[0]); fx[i] = f(xk[i])
        if max(fx) - min(fx) < tol:
            break
    return xk[best], fx[best]

def _hex_lattice():
    r = math.sqrt(3.0) / 2.0
    pts = [(-2, 0), (2, 0), (-1, 1), (1, 1), (-1, -1), (1, -1),
           (-3, 1), (3, 1), (-3, -1), (3, -1), (0, 2)]
    out = np.zeros((11, 3))
    for k, (q, rr) in enumerate(pts):
        out[k, 0] = 1.5 * q
        out[k, 1] = r * (rr + q / 2.0)
        out[k, 2] = 0.0
    return out

def _unpack(x):
    inner = x.reshape(11, 3)
    return inner, np.array([0.0, 0.0, 0.0])

def hexagon_packing_11():
    t0 = time.time()
    base = _hex_lattice()
    S0 = _outer_side(base)
    best_inner, best_outer, best_S = base.copy(), np.array([0.0, 0.0, 0.0]), S0
    rng = np.random.default_rng(12345)

    def objective(x):
        inner, outer = _unpack(x)
        S = _outer_side(inner)
        ok, _ = _valid(inner, outer, S)
        if not ok:
            return 1e8
        return 1.0 / S

    # local search from base
    x0 = base.reshape(-1)
    xb, fb = _nm(x0, objective)
    if fb < 1e7:
        bi, bo = _unpack(xb)
        S = _outer_side(bi)
        if S < best_S:
            best_inner, best_outer, best_S = bi.copy(), bo.copy(), S

    # jittered restarts
    for it in range(200):
        if time.time() > _DEADLINE - 30.0:
            break
        x0 = base.reshape(-1).copy()
        x0 += rng.normal(0.0, 0.45, x0.size)
        xb, fb = _nm(x0, objective)
        if fb < 1e7:
            bi, bo = _unpack(xb)
            S = _outer_side(bi)
            if S < best_S:
                best_inner, best_outer, best_S = bi.copy(), bo.copy(), S
        # also try a fresh small random cluster occasionally
        if it % 7 == 0:
            x0 = rng.normal(0.0, 0.8, 33)
            xb, fb = _nm(x0, objective)
            if fb < 1e7:
                bi, bo = _unpack(xb)
                S = _outer_side(bi)
                if S < best_S:
                    best_inner, best_outer, best_S = bi.copy(), bo.copy(), S

    if time.time() > _DEADLINE - 20.0:
        return best_inner, best_outer, float(best_S)
    return best_inner, best_outer, float(best_S)

# EVOLVE-BLOCK-END
