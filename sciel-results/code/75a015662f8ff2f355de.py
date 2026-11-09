import time
import math
import numpy as np

N = 11

_R = 1.0
_SQRT3 = math.sqrt(3.0)
_EPS = 1e-9

_VERT = np.array([
    [_R * math.cos(math.radians(a)), _R * math.sin(math.radians(a))]
    for a in (0, 60, 120, 180, 240, 300)
], dtype=float)
_EDGE_NORMAL = np.array([
    np.array([math.cos(math.radians(a)), math.sin(math.radians(a))])
    for a in (0, 60, 120, 180, 240, 300)
], dtype=float)
_EDGE_PAIRS = np.array([(0,1),(1,2),(2,3),(3,4),(4,5),(5,0)], dtype=int)


def _rot_matrix(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def _hex_vertices(cx, cy, theta):
    R = _rot_matrix(theta)
    return _VERT @ R.T + np.array([cx, cy])


def _min_edge_dist(A, B):
    best = float('inf')
    for i, j in _EDGE_PAIRS:
        p1 = A[i]; p2 = A[j]
        d = p2 - p1
        L = math.hypot(d[0], d[1]) + _EPS
        n = np.array([d[1] / L, -d[0] / L])
        for q in B:
            dist = float(n @ (q - p1))
            if dist < best:
                best = dist
    for i, j in _EDGE_PAIRS:
        p1 = B[i]; p2 = B[j]
        d = p2 - p1
        L = math.hypot(d[0], d[1]) + _EPS
        n = np.array([d[1] / L, -d[0] / L])
        for q in A:
            dist = float(n @ (q - p1))
            if dist < best:
                best = dist
    return best


def _enclosing_side(A):
    h = 0.0
    for k in range(6):
        n = _EDGE_NORMAL[k]
        s = float(np.max(A @ n))
        if s > h:
            h = s
    return h


def _smooth(x, k=20.0):
    z = k * x
    if z > 30:
        return x
    if z < -30:
        return 0.0
    return x / (1.0 + math.exp(-z))


def _build_A(cx, cy, th):
    return np.array([_hex_vertices(cx[i], cy[i], th[i]) for i in range(N)])


def _objective(params):
    cx = params[0::3]
    cy = params[1::3]
    th = params[2::3]
    A = _build_A(cx, cy, th)
    pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = _min_edge_dist(A[i], A[j])
            if d < 0:
                pen += _smooth(-d, k=25.0)
    side = _enclosing_side(A)
    if side < _SQRT3:
        side += _smooth(_SQRT3 - side, k=25.0)
    return side + 20.0 * pen


def _run():
    import scipy.optimize as opt
    deadline = time.time() + 110.0

    best = None
    best_side = float('inf')
    rng = np.random.default_rng(12345)

    def try_start(params0):
        nonlocal best, best_side
        if time.time() > deadline:
            return
        try:
            res = opt.minimize(_objective, params0, method='L-BFGS-B',
                               options={'maxiter': 150, 'maxls': 40,
                                        'ftol': 1e-8, 'gtol': 1e-6})
        except Exception:
            return
        if time.time() > deadline:
            return
        if res.fun < best_side:
            cx = res.x[0::3]; cy = res.x[1::3]; th = res.x[2::3]
            A = _build_A(cx, cy, th)
            ok = True
            for i in range(N):
                for j in range(i + 1, N):
                    if _min_edge_dist(A[i], A[j]) < -1e-4:
                        ok = False
            if ok:
                side = _enclosing_side(A)
                if side < best_side:
                    best_side = side
                    best = (cx.copy(), cy.copy(), th.copy())

    w = _SQRT3
    hex_pos = [(0.0, 0.0)]
    for k in range(6):
        ang = math.radians(60 * k + 30)
        hex_pos.append((w * math.cos(ang), w * math.sin(ang)))
    for k in range(4):
        ang = math.radians(60 * k)
        hex_pos.append((2 * w * math.cos(ang), 2 * w * math.sin(ang)))
    p0 = []
    for (x, y) in hex_pos:
        p0 += [x, y, 0.0]
    try_start(np.array(p0, dtype=float))

    for s in range(40):
        if time.time() > deadline:
            break
        cx = rng.uniform(-4, 4, size=N)
        cy = rng.uniform(-4, 4, size=N)
        th = rng.uniform(0, math.pi / 3, size=N)
        p = np.empty(3 * N)
        p[0::3] = cx
        p[1::3] = cy
        p[2::3] = th
        try_start(p)

    if best is None:
        cx = np.array([p[0] for p in hex_pos])
        cy = np.array([p[1] for p in hex_pos])
        th = np.zeros(N)
        A = _build_A(cx, cy, th)
        best = (cx, cy, th)
        best_side = _enclosing_side(A)

    cx, cy, th = best
    inner = [[float(cx[i]), float(cy[i]), float(math.degrees(th[i] % (math.pi / 3)))] for i in range(N)]
    return inner, (0.0, 0.0), float(best_side), 0.0


def construct():
    try:
        return _run()
    except Exception:
        w = _SQRT3
        hex_pos = [(0.0, 0.0)]
        for k in range(6):
            ang = math.radians(60 * k + 30)
            hex_pos.append((w * math.cos(ang), w * math.sin(ang)))
        for k in range(4):
            ang = math.radians(60 * k)
            hex_pos.append((2 * w * math.cos(ang), 2 * w * math.sin(ang)))
        inner = [[float(x), float(y), 0.0] for (x, y) in hex_pos]
        A = np.array([_hex_vertices(x, y, 0.0) for (x, y) in hex_pos])
        side = _enclosing_side(A)
        return inner, (0.0, 0.0), float(side), 0.0
