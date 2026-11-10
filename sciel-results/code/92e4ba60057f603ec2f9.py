# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = time.time() + 590.0  # safety margin before 630s kill

def _time_left():
    return _DEADLINE - time.time()

def _hex_verts(cx, cy, d, ang_deg):
    a = np.deg2rad(ang_deg)
    th = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack((cx + d * np.cos(th), cy + d * np.sin(th)))

def _outer_verts(c, d, ang_deg):
    return _hex_verts(c[0], c[1], d, ang_deg)

def _min_side_for_fixed(c, outer_c, outer_ang):
    """Minimum outer side (fixed inner config c, outer center c, angle) so all inner vertices inside."""
    V = _outer_verts(outer_c, 1.0, outer_ang)  # unit-scale outer
    V1 = np.roll(V, -1, axis=0)
    e = V1 - V
    L = np.linalg.norm(e, axis=1)
    if np.any(L <= 0):
        return np.inf
    nn = np.array([[-e[:, 1]], [e[:, 0]]]).T / L[:, None]
    h = nn @ (V - outer_c)
    if np.any(h <= 0):
        return np.inf
    req = np.zeros(6)
    for j in range(6):
        n = nn[j]
        pts = []
        for (ic, ia) in c:
            pts.append(_hex_verts(ic[0], ic[1], 1.0, ia))
        pts = np.vstack(pts)
        val = n @ (pts - outer_c)
        req[j] = val.max()
    d_need = req / h
    return float(np.max(d_need))

def _check_separation(c):
    """SAT on edge-normal axes of each unit hexagon."""
    hexes = [_hex_verts(ic[0], ic[1], 1.0, ia) for (ic, ia) in c]
    n = len(hexes)
    for i in range(n):
        V = hexes[i]
        V1 = np.roll(V, -1, axis=0)
        e = V1 - V
        L = np.linalg.norm(e, axis=1)
        nn = np.array([[-e[:, 1]], [e[:, 0]]]).T / L[:, None]
        for j in range(n):
            if i == j:
                continue
            W = hexes[j]
            for nvec in nn:
                pi = nvec @ V
                pj = nvec @ W
                gap = max(pi.min(), pj.min()) - min(pi.max(), pj.max())
                if gap < -1e-6:
                    return False
    return True

def _check_containment(c, d, outer_c, outer_ang):
    V = _outer_verts(outer_c, d, outer_ang)
    V1 = np.roll(V, -1, axis=0)
    e = V1 - V
    for (ic, ia) in c:
        pts = _hex_verts(ic[0], ic[1], 1.0, ia)
        for v in pts:
            for k in range(6):
                p1 = V[k]
                cr = e[k][0] * (v[1] - p1[1]) - e[k][1] * (v[0] - p1[0])
                if cr < -1e-6:
                    return False
    return True

def _valid(c, d, outer_c, outer_ang):
    if not np.isfinite(d) or d < np.sqrt(12.0) - 1e-9:
        return False
    if not _check_separation(c):
        return False
    if not _check_containment(c, d, outer_c, outer_ang):
        return False
    return True

def _analytic_seed():
    h = np.sqrt(3.0) / 2.0
    centers = [
        (0.0, 0.0),
        (1.0, h), (-1.0, h), (0.0, 2.0 * h),
        (1.0, -h), (-1.0, -h), (0.0, -2.0 * h),
        (2.0, 2.0 * h), (-2.0, 2.0 * h),
        (2.0, -2.0 * h), (-2.0, -2.0 * h),
        (0.0, 0.0),
    ]
    seen = set()
    out = []
    for (x, y) in centers:
        key = (round(x, 6), round(y, 6))
        if key in seen:
            continue
        seen.add(key)
        out.append((x, y))
    extra = [(3.0, h), (-3.0, h), (3.0, -h), (-3.0, -h), (0.0, 3.0 * h), (0.0, -3.0 * h)]
    for (x, y) in extra:
        if len(out) >= 12:
            break
        key = (round(x, 6), round(y, 6))
        if key not in seen:
            seen.add(key)
            out.append((x, y))
    out = out[:12]
    c = [(np.array([x, y]), 0.0) for (x, y) in out]
    outer_c = np.array([0.0, 0.0])
    outer_ang = 0.0
    d = _min_side_for_fixed(c, outer_c, outer_ang)
    return c, d, outer_c, outer_ang

def _nelder_mead(f, x0, max_iter=2000, step=0.05, tol=1e-7):
    n = len(x0)
    x = np.array(x0, dtype=float)
    S = [x.copy()]
    for i in range(n):
        xi = x.copy()
        xi[i] += step if x[i] == 0 else 0.5 * x[i]
        S.append(xi)
    for _ in range(max_iter):
        if _time_left() < 5.0:
            break
        vals = [f(s) for s in S]
        order = np.argsort(vals)
        S = [S[i] for i in order]
        vals = [vals[i] for i in order]
        if np.isfinite(vals[0]) and abs(vals[-1] - vals[0]) < tol:
            break
        centroid = np.mean(S[:-1], axis=0)
        xr = centroid + (centroid - S[-1])
        fr = f(xr)
        if fr < vals[0]:
            xe = centroid + 2.0 * (centroid - S[-1])
            fe = f(xe)
            S[-1] = xe if fe < fr else xr
        elif fr < vals[-2]:
            S[-1] = xr
        else:
            xc = centroid + 0.5 * (S[-1] - centroid)
            fc = f(xc)
            if fc < vals[-1]:
                S[-1] = xc
            else:
                for i in range(1, n + 1):
                    S[i] = 0.5 * (S[i] + S[0])
    return min(S, key=lambda s: f(s))

def hexagon_packing_12():
    c, d, outer_c, outer_ang = _analytic_seed()

    def encode(c, outer_c, outer_ang):
        v = []
        for (ic, ia) in c:
            v.append(ic[0]); v.append(ic[1]); v.append(ia)
        v.append(outer_c[0]); v.append(outer_c[1]); v.append(outer_ang)
        return np.array(v)

    def decode(v):
        c = []
        for i in range(12):
            c.append((np.array([v[3*i], v[3*i+1]]), v[3*i+2]))
        outer_c = np.array([v[36], v[37]])
        outer_ang = v[38]
        return c, outer_c, outer_ang

    def objective(v):
        c, outer_c, outer_ang = decode(v)
        if not _check_separation(c):
            return 1e6 + 1000.0 * max(0.0, np.max(np.abs(v)) - 10.0)
        d = _min_side_for_fixed(c, outer_c, outer_ang)
        if not np.isfinite(d) or d < np.sqrt(12.0) - 1e-9:
            return 1e6
        if not _check_containment(c, d, outer_c, outer_ang):
            return 1e6
        return d

    v0 = encode(c, outer_c, outer_ang)
    best_v = v0.copy()
    best_d = objective(v0)

    if _time_left() > 30.0:
        v1 = _nelder_mead(objective, v0, max_iter=1500, step=0.03, tol=1e-8)
        d1 = objective(v1)
        if d1 < best_d:
            best_v, best_d = v1, d1

    rng = np.random.default_rng(12345)
    for _ in range(3):
        if _time_left() < 20.0:
            break
        v0r = v0 + rng.normal(0.0, 0.15, size=len(v0))
        vr = _nelder_mead(objective, v0r, max_iter=800, step=0.05, tol=1e-8)
        dr = objective(vr)
        if dr < best_d:
            best_v, best_d = vr, dr

    c, outer_c, outer_ang = decode(best_v)
    d_final = _min_side_for_fixed(c, outer_c, outer_ang)
    if not _valid(c, d_final, outer_c, outer_ang):
        c, d_final, outer_c, outer_ang = _analytic_seed()

    inner_hex_data = np.array([[ic[0][0], ic[0][1], ia] for (ic, ia) in c], dtype=float)
    outer_hex_data = np.array([outer_c[0], outer_c[1], outer_ang], dtype=float)
    return inner_hex_data, outer_hex_data, float(d_final)

# EVOLVE-BLOCK-END
