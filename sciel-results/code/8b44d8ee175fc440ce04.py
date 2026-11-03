import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 6.0
EPS = 1e-12

def _time_ok():
    return (time.time() - START_TIME) < (TIME_LIMIT - SAFETY_MARGIN)

def _hex_normals(theta):
    # 6 outward face normals of a regular hexagon with "orientation" theta
    return np.array([[math.cos(theta + math.pi / 6.0 + k * math.pi / 3.0),
                      math.sin(theta + math.pi / 6.0 + k * math.pi / 3.0)] for k in range(6)])

def _softmax(arr, t):
    a = np.asarray(arr)
    m = a.max()
    if not np.isfinite(m):
        return float('inf')
    e = np.exp(np.clip((a - m) / t, -50.0, 50.0))
    return m + t * math.log(e.sum())

def _outer_radius(pts, cx, cy, theta):
    # max over 6 face normals of max_i |dot(p_i - c, n)|
    N = _hex_normals(theta)
    D = pts - np.array([cx, cy])
    P = D @ N.T                       # (11, 6)
    return float(np.max(np.abs(P)))

def _separation(pts):
    # min over pairs of center distance
    n = pts.shape[0]
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return float(d.min())

def _objective(params, t, lam):
    # params: [x0,y0,th0, ..., x10,y10,th10, cx, cy, theta, R]
    n = N
    pos = params[:3 * n].reshape(n, 3)
    cx, cy, theta = params[3 * n], params[3 * n + 1], params[3 * n + 2]
    R = params[3 * n + 3]
    pts = pos[:, :2]
    R = max(R, 0.5)
    # outer containment penalty (smooth)
    outer = _outer_radius(pts, cx, cy, theta)
    pen_outer = _softmax(np.array([outer - R]), t)
    # inner non-overlap penalty (smooth): want separation >= 2
    sep = _separation(pts)
    pen_inner = _softmax(np.array([2.0 - sep]), t)
    return pen_outer + lam * pen_inner + 0.5 * R

def _evaluate(pts, cx, cy, theta):
    # exact enclosing side length for a given config
    R = _outer_radius(pts, cx, cy, theta)
    sep = _separation(pts)
    return (R + APOTHEM) / APOTHEM, sep

def _seed_ring():
    # center + ring of 10 around it (loose)
    pts = [np.array([0.0, 0.0])]
    for k in range(10):
        a = 2 * math.pi * k / 10.0
        pts.append(np.array([2.3 * math.cos(a), 2.3 * math.sin(a)]))
    thetas = [0.0] * N
    cx, cy, theta = 0.0, 0.0, 0.0
    R = _outer_radius(np.array(pts), cx, cy, theta) + 0.2
    params = []
    for (x, y), th in zip(pts, thetas):
        params += [x, y, th]
    params += [cx, cy, theta, R]
    return np.array(params, dtype=float)

def _seed_lattice():
    # hexagonal lattice patch
    pts = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append([x, y])
    pts = pts[:N]
    thetas = [0.0] * N
    cx, cy, theta = 0.0, 0.0, 0.0
    R = _outer_radius(np.array(pts), cx, cy, theta) + 0.2
    params = []
    for (x, y), th in zip(pts, thetas):
        params += [x, y, th]
    params += [cx, cy, theta, R]
    return np.array(params, dtype=float)

def _seed_random(rng):
    pts = []
    for i in range(N):
        a = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(0, 3.0)
        pts.append([r * math.cos(a), r * math.sin(a)])
    thetas = [rng.uniform(0, math.pi / 3.0) for _ in range(N)]
    cx, cy, theta = 0.0, 0.0, rng.uniform(0, math.pi / 6.0)
    R = _outer_radius(np.array(pts), cx, cy, theta) + 0.2
    params = []
    for (x, y), th in zip(pts, thetas):
        params += [x, y, th]
    params += [cx, cy, theta, R]
    return np.array(params, dtype=float)

def _anneal(params0, rng):
    # schedule of (temperature, lam)
    schedule = [
        (0.6, 3.0), (0.3, 4.0), (0.15, 5.0), (0.07, 6.0),
        (0.035, 7.0), (0.015, 8.0), (0.006, 9.0), (0.002, 10.0),
    ]
    cur = params0.copy()
    for t, lam in schedule:
        if not _time_ok():
            break
        def obj(p, _t=t, _lam=lam):
            return _objective(p, _t, _lam)
        try:
            res = minimize(obj, cur, method='Nelder-Mead',
                           options={'maxiter': 1200, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < obj(cur):
                cur = res.x
        except Exception:
            pass
    return cur

def construct():
    if not _time_ok():
        p = _seed_lattice()
        pts = p[:3 * N].reshape(N, 3)[:, :2]
        cx, cy, theta = p[3 * N], p[3 * N + 1], p[3 * N + 2]
        S, _ = _evaluate(pts, cx, cy, theta)
        inner = [[float(x), float(y), float(math.degrees(th))] for (x, y, th) in p[:3 * N].reshape(N, 3)]
        return inner, [float(cx), float(cy)], float(S), float(math.degrees(theta))

    rng = np.random.default_rng(12345)
    seeds = [_seed_ring(), _seed_lattice()]
    for i in range(6):
        seeds.append(_seed_random(rng))

    best_S = float('inf')
    best_cfg = None

    for s in seeds:
        if not _time_ok():
            break
        cur = _anneal(s, rng)
        pts = cur[:3 * N].reshape(N, 3)[:, :2]
        thetas = cur[:3 * N].reshape(N, 3)[:, 2]
        cx, cy, theta = cur[3 * N], cur[3 * N + 1], cur[3 * N + 2]
        # polish the enclosing hexagon tightly for this frozen config
        def outer_obj(v):
            cx2, cy2, th2 = v
            return _outer_radius(pts, cx2, cy2, th2)
        try:
            res = minimize(outer_obj, [cx, cy, theta], method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
            cx, cy, theta = res.x
        except Exception:
            pass
        S, sep = _evaluate(pts, cx, cy, theta)
        if S < best_S and sep >= 2.0 - 1e-6:
            best_S = S
            best_cfg = (pts.copy(), thetas.copy(), cx, cy, theta)

    if best_cfg is None:
        # fallback: lattice config
        p = _seed_lattice()
        pts = p[:3 * N].reshape(N, 3)[:, :2]
        thetas = p[:3 * N].reshape(N, 3)[:, 2]
        cx, cy, theta = p[3 * N], p[3 * N + 1], p[3 * N + 2]
        S, _ = _evaluate(pts, cx, cy, theta)
        best_S, best_cfg = S, (pts, thetas, cx, cy, theta)

    pts, thetas, cx, cy, theta = best_cfg
    # tiny safety shrink of positions about center to guarantee no overlap
    SCALE = 1.0 - 1e-9
    pts = np.array([cx + (x - cx) * SCALE, cy + (y - cy) * SCALE]).T

    R = _outer_radius(pts, cx, cy, theta)
    S = (R + APOTHEM) / APOTHEM

    inner = [[float(pts[i, 0]), float(pts[i, 1]), float(math.degrees(thetas[i]))] for i in range(N)]
    return inner, [float(cx), float(cy)], float(S), float(math.degrees(theta))
