"""Hexagon packing: minimize enclosing regular-hexagon side length."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11
T0 = time.time()
DEADLINE = 150.0
MARGIN = 15.0          # stop this many seconds before the kill
SMOOTH = 0.02          # smoothness scale for overlap / containment penalties


def _hex_verts(cx, cy, ang):
    a = np.deg2rad(ang)
    c, s = np.cos(a), np.sin(a)
    base = np.array([[np.cos(np.deg2rad(k * 60)), np.sin(np.deg2rad(k * 60))] for k in range(6)])
    R = np.array([[c, -s], [s, c]])
    return np.array([cx, cy]) + base @ R.T


def _out_verts(cx, cy, S, ang):
    a = np.deg2rad(ang)
    c, s = np.cos(a), np.sin(a)
    base = np.array([[np.cos(np.deg2rad(k * 60)), np.sin(np.deg2rad(k * 60))] for k in range(6)]) * S
    R = np.array([[c, -s], [s, c]])
    return np.array([cx, cy]) + base @ R.T


def _proj_interval(P, d):
    t = P @ d
    return t.min(), t.max()


def _overlap_iou(P, Q):
    """Smoothed SAT overlap of two convex hexagons (0..1)."""
    min_ov = np.inf
    min_area = 0.0
    for poly in (P, Q):
        for i in range(6):
            p1 = poly[i]
            p2 = poly[(i + 1) % 6]
            e = p2 - p1
            L = np.linalg.norm(e)
            if L < 1e-12:
                continue
            n = np.array([-e[1], e[0]])
            n /= np.linalg.norm(n)
            i1, i2 = _proj_interval(P, n)
            j1, j2 = _proj_interval(Q, n)
            ov = min(i2, j2) - max(i1, j1)
            if ov < 0:
                return 0.0
            min_ov = min(min_ov, ov)
            min_area = max(min_area, L)
    if min_ov <= 0:
        return 0.0
    # smooth: saturate near 0, scale by overlap relative to edge length
    x = min_ov / max(min_area, 1e-9)
    return 1.0 - np.exp(-(x / SMOOTH))


def _max_viol(P, Q):
    """Smoothed max penetration depth of P's vertices into Q (0..1)."""
    viol = 0.0
    for i in range(6):
        p = P[i]
        for j in range(6):
            a = Q[j]
            b = Q[(j + 1) % 6]
            ab = b - a
            L2 = float(ab @ ab)
            if L2 < 1e-12:
                continue
            t = np.clip(float((p - a) @ ab) / L2, 0.0, 1.0)
            cp = a + t * ab
            d = p - cp
            dist = np.linalg.norm(d)
            n = d / dist if dist > 1e-12 else np.array([1.0, 0.0])
            # inward normal of edge a->b (Q CCW)
            en = np.array([-(b[1] - a[1]), b[0] - a[0]])
            en /= np.linalg.norm(en)
            # signed: negative if p is inside relative to this edge
            s = float((p - a) @ en)
            if s < 0:
                viol = max(viol, -s)
    x = viol / SMOOTH
    return 1.0 - np.exp(-x)


def _obj(p):
    # p = [S, ox, oy, oa, x1,y1,a1, ... x11,y11,a11]
    S = p[0]
    ox, oy, oa = p[1], p[2], p[3]
    O = _out_verts(ox, oy, S, oa)
    inner = p[4:].reshape(N, 3)
    # containment: each inner vertex must be inside outer
    cont = 0.0
    for k in range(N):
        P = _hex_verts(inner[k, 0], inner[k, 1], inner[k, 2])
        for i in range(6):
            pv = P[i]
            for j in range(6):
                a = O[j]
                b = O[(j + 1) % 6]
                ab = b - a
                L2 = float(ab @ ab)
                if L2 < 1e-12:
                    continue
                t = np.clip(float((pv - a) @ ab) / L2, 0.0, 1.0)
                cp = a + t * ab
                d = pv - cp
                dist = np.linalg.norm(d)
                n = d / dist if dist > 1e-12 else np.array([1.0, 0.0])
                en = np.array([-(b[1] - a[1]), b[0] - a[0]])
                en /= np.linalg.norm(en)
                s = float((pv - a) @ en)
                if s < 0:
                    cont += 1.0 - np.exp((-s / SMOOTH))
    # overlap
    ov = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            Pi = _hex_verts(inner[i, 0], inner[i, 1], inner[i, 2])
            Pj = _hex_verts(inner[j, 0], inner[j, 1], inner[j, 2])
            ov += _max_viol(Pi, Pj)
    return S + 10.0 * ov + 10.0 * cont


def _validity(p):
    """Return (is_valid, true_side) using hard geometry checks."""
    S = p[0]
    ox, oy, oa = p[1], p[2], p[3]
    O = _out_verts(ox, oy, S, oa)
    inner = p[4:].reshape(N, 3)
    verts = [_hex_verts(inner[k, 0], inner[k, 1], inner[k, 2]) for k in range(N)]
    # containment
    for P in verts:
        for i in range(6):
            pv = P[i]
            ok = True
            for j in range(6):
                a = O[j]
                b = O[(j + 1) % 6]
                cross = (b[0] - a[0]) * (pv[1] - a[1]) - (b[1] - a[1]) * (pv[0] - a[0])
                if cross < -1e-7:
                    ok = False
                    break
            if not ok:
                return False, None
    # overlap
    for i in range(N):
        for j in range(i + 1, N):
            if _overlap_iou(verts[i], verts[j]) > 1e-9:
                return False, None
    return True, S


def _init_topologies():
    inits = []
    # 1) hexagonal cluster: 1 center + 6 ring + 4 more
    d = 2.0  # center distance for touching hexes
    cx, cy = 0.0, 0.0
    pts = [(cx, cy, 0.0)]
    for k in range(6):
        a = np.deg2rad(k * 60)
        pts.append((cx + d * np.cos(a), cy + d * np.sin(a), 0.0))
    # 4 more around
    for k in range(4):
        a = np.deg2rad(30 + k * 90)
        pts.append((cx + 2 * d * np.cos(a), cy + 2 * d * np.sin(a), 0.0))
    inits.append(pts)
    # 2) linear
    pts = [(2.0 * k, 0.0, 0.0) for k in range(N)]
    inits.append(pts)
    # 3) two rows
    pts = []
    for k in range(6):
        pts.append((2.0 * k, 0.0, 0.0))
    for k in range(5):
        pts.append((2.0 * k + 1.0, 1.7320508, 0.0))
    inits.append(pts)
    # 4) circular
    for k in range(N):
        a = np.deg2rad(k * 360 / N)
        inits.append and None
    circ = [(3.0 * np.cos(np.deg2rad(k * 360 / N)), 3.0 * np.sin(np.deg2rad(k * 360 / N)), 0.0) for k in range(N)]
    inits.append(circ)
    # 5) random
    rng = np.random.default_rng(0)
    rpts = [(rng.uniform(-4, 4), rng.uniform(-4, 4), rng.uniform(0, 60)) for _ in range(N)]
    inits.append(rpts)
    return inits


def _initial_S(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    cx = (max(xs) + min(xs)) / 2
    cy = (max(ys) + min(ys)) / 2
    # bounding radius
    r = max(np.hypot(p[0] - cx, p[1] - cy) for p in pts) + 1.0
    return r * 1.2 + 1.0, cx, cy


def construct():
    best = None
    best_score = np.inf
    topologies = _init_topologies()
    for top in topologies:
        if time.time() > DEADLINE - MARGIN:
            break
        S0, cx0, cy0 = _initial_S(top)
        p0 = np.array([S0, cx0, cy0, 0.0] + [c for pt in top for c in pt], dtype=float)
        bounds = [(0.5, 50.0), (-50, 50), (-50, 50), (0.0, 60.0)]
        for _ in range(N):
            bounds += [(-50, 50), (-50, 50), (0.0, 60.0)]
        try:
            res = minimize(_obj, p0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 300, 'maxfun': 4000, 'gtol': 1e-7})
        except Exception:
            continue
        p = res.x
        ok, side = _validity(p)
        if ok and side < best_score:
            best_score = side
            best = (p, side)
        # try a quick second refinement from best found so far
    if best is None:
        # fallback to seed-like layout
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    p, side = best
    inner = [[float(p[4 + 3 * k]), float(p[5 + 3 * k]), float(p[6 + 3 * k])] for k in range(N)]
    return inner, (float(p[1]), float(p[2])), float(p[0]), float(p[3])


if __name__ == '__main__':
    r = construct()
    print("side:", r[2])
