"""Pack 11 unit regular hexagons into the smallest enclosing regular hexagon."""
import time
import math
import numpy as np

N = 11
T0 = time.time()
DEADLINE = 150.0
MARGIN = 20.0          # safety margin before the 150 s kill
SMOOTH = 0.02          # soft-min / soft-max temperature

# unit regular hexagon, flat-top, circumradius = side = 1
_BASE = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))]
                  for a in range(0, 360, 60)])


def _rot(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def _inner_verts(pos, ang):
    return pos + _BASE @ _rot(ang).T


def _outer_verts(cx, cy, s, phi):
    # regular hexagon, side s (== circumradius), rotated by phi
    return np.array([[cx + s * math.cos(phi + math.radians(a)),
                      cy + s * math.sin(phi + math.radians(a))]
                     for a in range(0, 360, 60)])


def _halfwidths(verts, phi):
    # 6 inward half-widths (support values) of a convex polygon for the
    # outer hexagon's face normals (normals of outer hexagon rotated by phi)
    angs = phi + np.radians(np.arange(6) * 60.0)
    n = np.stack([np.cos(angs), np.sin(angs)], axis=1)
    return np.max(verts @ n.T, axis=1)


def _outer_side(inner, phi):
    # exact smallest enclosing regular hexagon of orientation phi
    if len(inner) == 0:
        return 0.0
    allv = np.vstack(inner)
    c = 0.5 * (allv.max(axis=0) + allv.min(axis=0))
    d = allv - c
    hw = _halfwidths(d, phi)
    return float(np.max(hw))


def _overlap(a, b):
    # smooth non-overlap: 0 if separated, ~soft max(0, -min gap) otherwise
    min_gap = 1.0
    for va in a:
        for vb in b:
            min_gap = min(min_gap, np.linalg.norm(va - vb))
    return -SMOOTH * math.log(1.0 + math.exp(min_gap / SMOOTH))


def _outer_penalty(inner, cx, cy, s, phi):
    # smooth containment: soft-min over all inner verts of (halfwidth - s)
    ov = _outer_verts(cx, cy, s, phi)
    c = 0.5 * (ov.max(axis=0) + ov.min(axis=0))
    d = ov - c
    hw = _halfwidths(d, phi)
    over = s - hw                       # positive => inside
    min_over = over.min()
    return -SMOOTH * math.log(1.0 + math.exp(min_over / SMOOTH))


def _objective(x):
    s = x[0]
    phi = x[1]
    pos = x[2:2 + 2 * N].reshape(N, 2)
    ang = x[2 + 2 * N:2 + 2 * N + N]
    inner = [_inner_verts(pos[i], ang[i]) for i in range(N)]
    val = s
    val += 1000.0 * _outer_penalty(inner, 0.0, 0.0, s, phi)
    for i in range(N):
        for j in range(i + 1, N):
            val += 1000.0 * _overlap(inner[i], inner[j])
    return val


def _solve(x0):
    from scipy.optimize import minimize
    res = minimize(_objective, x0, method='L-BFGS-B',
                   options={'maxiter': 400, 'maxfun': 40000, 'ftol': 1e-12})
    s = float(res.x[0])
    phi = float(res.x[1])
    pos = res.x[2:2 + 2 * N].reshape(N, 2)
    ang = res.x[2 + 2 * N:2 + 2 * N + N]
    inner = [_inner_verts(pos[i], ang[i]) for i in range(N)]
    exact = _outer_side(inner, phi)
    return max(exact, s), pos, ang, phi


def construct():
    best = (40.0, None, None, None)   # (side, pos, ang, phi)

    def consider(side, pos, ang, phi):
        nonlocal best
        if side < best[0] and np.isfinite(side):
            best = (side, pos, ang, phi)

    # ---- multi-start ----
    starts = [
        np.array([4.0 * k for k in range(-(N // 2), N - N // 2)]),  # row x
        np.zeros(N),                                                # row y
        np.zeros(N),                                                # shared ang
        0.0,                                                        # outer phi
    ]
    base = np.concatenate([starts])
    rng = np.random.default_rng(12345)

    attempts = 0
    while time.time() < DEADLINE - MARGIN:
        if attempts == 0:
            x0 = base.copy()
        else:
            x0 = base.copy()
            x0[2:] += rng.normal(0.0, 1.2, size=x0[2:].shape)
        attempts += 1
        try:
            side, pos, ang, phi = _solve(x0)
        except Exception:
            continue
        consider(side, pos, ang, phi)
        # stop early if we've clearly beaten the seed a lot and time is tight
        if side < 6.0 and time.time() > DEADLINE - 2 * MARGIN:
            break

    if best[1] is None:
        # fallback: the seed row
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # ---- final high-precision polish (tighter containment, no overlap) ----
    side, pos, ang, phi = best
    if time.time() < DEADLINE - 4.0:
        global SMOOTH
        saved = SMOOTH
        SMOOTH = 0.005
        try:
            x0 = np.concatenate([[side, phi], pos.ravel(), ang])
            side2, pos2, ang2, phi2 = _solve(x0)
            consider(side2, pos2, ang2, phi2)
        except Exception:
            pass
        finally:
            SMOOTH = saved

    side, pos, ang, phi = best
    inner = [[float(pos[i][0]), float(pos[i][1]),
              float(math.degrees(ang[i]))] for i in range(N)]
    return inner, (0.0, 0.0), float(side), float(math.degrees(phi))
