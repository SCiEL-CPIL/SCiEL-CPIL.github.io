import time
import numpy as np

N = 12
DEADLINE = 140.0  # safety margin under 150s wall-clock

def _hex_verts(cx, cy, r, theta):
    a = np.deg2rad(theta)
    c, s = np.cos(a), np.sin(a)
    base = np.array([[np.cos(np.deg2rad(60 * k)), np.sin(np.deg2rad(60 * k))] for k in range(6)])
    R = np.array([[c, -s], [s, c]])
    return r * (base @ R.T) + np.array([cx, cy])

def _poly_sep(A, B):
    # minimum translation distance to separate convex polys A,B (0 if disjoint).
    n = len(A)
    best = 0.0
    for i in range(n):
        p1 = A[i]; p2 = A[(i + 1) % n]
        e = p2 - p1
        nrm = np.array([-e[1], e[0]])
        nn = np.linalg.norm(nrm)
        if nn == 0:
            continue
        nrm = nrm / nn
        dA = A @ nrm
        dB = B @ nrm
        loA = dA.min(); hiA = dA.max()
        loB = dB.min(); hiB = dB.max()
        if hiA < loB:
            continue
        if hiB < loA:
            continue
        gap = max(loB - hiA, loA - hiB)
        if gap > best:
            best = gap
    return best

def _pair_pen(A, B):
    d = _poly_sep(A, B)
    return 0.0 if d <= 1e-9 else (1.0 - d) ** 2

def _enclosing(params):
    # params: [cx, cy, theta_deg, side]; returns minimal enclosing side length
    cx, cy, th, side = params
    a = np.deg2rad(th)
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c, -s], [s, c]])
    # unit hexagon (circumradius 1) vertices
    base = np.array([[np.cos(np.deg2rad(60 * k)), np.sin(np.deg2rad(60 * k))] for k in range(6)])
    verts = base @ R.T
    # support function of enclosing hexagon H(S) in direction u: S * (u·v_max)
    # For a regular hexagon of circumradius S with orientation R, support in dir u = S * max_k (u·verts_k)
    # We need S >= max over all points p, all dirs u of (u·(p-c)) / max_k(u·verts_k)
    # Use the 6 face normals of the unit hexagon as candidate directions (tight for hexagon containment)
    normals = []
    for k in range(6):
        e = verts[(k + 1) % 6] - verts[k]
        nn = np.array([-e[1], e[0]])
        nn = nn / np.linalg.norm(nn)
        normals.append(nn)
    normals = np.array(normals)
    bestS = 0.0
    for u in normals:
        # support of H(S) in dir u
        supH = max((u @ v) for v in verts)
        if supH <= 0:
            continue
        # required S so that H(S) contains all points
        for p in _ALL_POINTS:
            val = (u @ (p - np.array([cx, cy]))) / supH
            if val > bestS:
                bestS = val
    return bestS

_ALL_POINTS = None  # filled per evaluation

def _total_penalty(polys):
    pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            pen += _pair_pen(polys[i], polys[j])
    return pen

def construct():
    global _ALL_POINTS
    t0 = time.time()

    def now():
        return time.time()

    def timed():
        return now() - t0 > DEADLINE

    # Initial: ring of 12 hexagons around origin, plus a center one would be 13; use 12 in a ring
    # 12 hexagons: 6 around center at radius ~2, 6 offset -> actually just place in a 3x4 grid for validity
    # Start from a valid grid packing
    rng = np.random.default_rng(0)
    # grid packing: 4 cols x 3 rows of hexagons (flat-top), spacing
    # For flat-top unit hexagon, horizontal spacing 3, vertical spacing 2*sqrt(3)
    pos = []
    for r in range(3):
        for c in range(4):
            x = 3.0 * (c - 1.5)
            y = 2 * np.sqrt(3) * (r - 1.0)
            pos.append([x, y, 0.0])
    pos = np.array(pos)

    best = None
    best_val = np.inf

    def evaluate(p):
        # p: (N,3) [x,y,theta_deg]
        polys = [_hex_verts(p[i, 0], p[i, 1], 1.0, p[i, 2]) for i in range(N)]
        _ALL_POINTS = np.vstack(polys)
        pen = _total_penalty(polys)
        if pen > 1e-7:
            return np.inf, pen
        # minimize enclosing side over (cx, cy, theta, side) via simple search
        # center = centroid
        cx0, cy0 = _ALL_POINTS[:, 0].mean(), _ALL_POINTS[:, 1].mean()
        bestS = np.inf
        for th in range(0, 30, 5):
            S = _enclosing([cx0, cy0, float(th), 1.0])
            if S < bestS:
                bestS = S
        return bestS, pen

    val0, pen0 = evaluate(pos)
    if val0 < best_val:
        best_val = val0
        best = pos.copy()

    # Local search: perturb positions and angles
    step = 0.5
    iters = 0
    max_iters = 20000
    while iters < max_iters and not timed():
        iters += 1
        p = best + rng.normal(0, step, size=best.shape)
        val, pen = evaluate(p)
        if val < best_val:
            best_val = val
            best = p.copy()
            step = min(step * 1.05, 2.0)
        else:
            step = max(step * 0.95, 0.02)

    # Final: refine enclosing orientation more finely
    polys = [_hex_verts(best[i, 0], best[i, 1], 1.0, best[i, 2]) for i in range(N)]
    _ALL_POINTS = np.vstack(polys)
    cx0, cy0 = _ALL_POINTS[:, 0].mean(), _ALL_POINTS[:, 1].mean()
    bestS = np.inf
    for th in np.arange(0, 30, 0.5):
        S = _enclosing([cx0, cy0, float(th), 1.0])
        if S < bestS:
            bestS = S

    inner = [[float(best[i, 0]), float(best[i, 1]), float(best[i, 2])] for i in range(N)]
    return inner, (float(cx0), float(cy0)), float(bestS), 0.0
