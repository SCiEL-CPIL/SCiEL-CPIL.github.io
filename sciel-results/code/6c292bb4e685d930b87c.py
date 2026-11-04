import time
import math
import numpy as np
from scipy.optimize import minimize

N = 12
_DEADLINE = time.time() + 140.0  # safety margin under 150s wall-clock

# 6 vertex directions for a unit regular hexagon (side length 1),
# parameterized by rotation. Support function of a unit hexagon.
def _hex_supports(theta):
    # outward normal directions (flat-top base at theta=0)
    base = np.array([0.0, 1.0, math.sqrt(3)/2.0, 0.0, -1.0, -math.sqrt(3)/2.0])
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return R @ base

def _hex_vertices(cx, cy, theta):
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]])
    base = np.array([[0.0, 1.0], [math.sqrt(3)/2.0, 0.5], [math.sqrt(3)/2.0, -0.5],
                     [0.0, -1.0], [-math.sqrt(3)/2.0, -0.5], [-math.sqrt(3)/2.0, 0.5]])
    return (R @ base.T).T + np.array([cx, cy])

def _separate(a, b, ta, tb):
    """max over pairs of outward normals of the gap between hex a (normal dir) and b."""
    sa = _hex_supports(ta)
    sb = _hex_supports(tb)
    gap = 0.0
    for k in range(6):
        d = sa[k]
        # support of a in direction d minus support of b in direction d
        # support_b = max_j (b_center + sb[j]) . d
        val = (a - b) @ d + (sb @ d).max()
        if val > gap:
            gap = val
    for k in range(6):
        d = sb[k]
        val = (a - b) @ d + (sa @ d).max()
        if val > gap:
            gap = val
    return gap

def _outer_side(cents, thetas, center):
    """minimum enclosing regular hexagon side length over outer orientation."""
    best = np.inf
    for phi in np.linspace(0.0, math.pi/3.0, 13):
        sup = _hex_supports(phi)
        r = 0.0
        for c, t in zip(cents, thetas):
            s = _hex_supports(t)
            for k in range(6):
                d = sup[k]
                val = (c - center) @ d + (s @ d).max()
                if val > r:
                    r = val
        # side length of enclosing hexagon whose support radius is r:
        # for a regular hexagon, circumradius = side, inradius = sqrt(3)/2 * side
        # support radius (inradius) = sqrt(3)/2 * side  =>  side = r / (sqrt(3)/2)
        side = r / (math.sqrt(3)/2.0)
        if side < best:
            best = side
    return best

def _objective(x):
    n = N
    cx = x[:n]; cy = x[n:2*n]; th = x[2*n:3*n]
    ocx = x[3*n]; ocy = x[3*n+1]
    cents = np.stack([cx, cy], axis=1)
    # overlap penalty
    pen = 0.0
    for i in range(n):
        for j in range(i+1, n):
            g = _separate(cents[i], cents[j], th[i], th[j])
            if g < 0.0:
                pen += g * g
    # containment: inner must be inside outer; use a generous outer side to bound
    outer = _outer_side(cents, th, np.array([ocx, ocy]))
    return outer + 40.0 * pen

def construct():
    global _DEADLINE
    rng = np.random.default_rng(12345)
    best = None
    best_val = np.inf

    # initial layout: ring of 6 + inner cluster of 6
    def make_start(seed):
        r = np.random.default_rng(seed)
        cx = np.zeros(N); cy = np.zeros(N); th = np.zeros(N)
        # outer ring of 6 around origin
        for k in range(6):
            ang = math.pi/3.0 * k
            rad = 2.0
            cx[k] = rad * math.cos(ang)
            cy[k] = rad * math.sin(ang)
        # inner 6 in a tight cluster
        for k in range(6):
            ang = math.pi/3.0 * k + math.pi/6.0
            rad = 1.1
            cx[6+k] = rad * math.cos(ang)
            cy[6+k] = rad * math.sin(ang)
        th[:] = r.uniform(-math.pi/6, math.pi/6, N)
        return cx, cy, th

    n_starts = 0
    for seed in range(60):
        if time.time() > _DEADLINE - 20.0:
            break
        cx, cy, th = make_start(seed)
        x0 = np.concatenate([cx, cy, th, [0.0, 0.0]])
        res = minimize(_objective, x0, method='L-BFGS-B',
                       options={'maxiter': 400, 'ftol': 1e-12})
        val = _objective(res.x)
        n_starts += 1
        if val < best_val:
            best_val = val
            best = res.x
        if best_val < 3.6:  # good enough, stop
            break

    if best is None:
        # fallback trivial
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N-N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    n = N
    cx = best[:n]; cy = best[n:2*n]; th = best[2*n:3*n]
    ocx = best[3*n]; ocy = best[3*n+1]
    cents = np.stack([cx, cy], axis=1)
    outer = _outer_side(cents, th, np.array([ocx, ocy]))
    inner = [[float(cx[i]), float(cy[i]), float(math.degrees(th[i]))] for i in range(n)]
    return inner, (float(ocx), float(ocy)), float(outer), 0.0
