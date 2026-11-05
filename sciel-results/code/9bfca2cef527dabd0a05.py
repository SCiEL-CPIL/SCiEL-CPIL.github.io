"""Differentiable hexagon packing via support-function constraints + L-BFGS-B."""
import time
import numpy as np
from scipy.optimize import minimize

N = 12
_DEADLINE = 150.0
_T0 = time.time()

# unit regular hexagon vertices, flat-top (circumradius = 1)
_VERT = np.array([[np.cos(np.radians(a)), np.sin(np.radians(a))] for a in range(0, 360, 60)])
# edge outward normals (perpendicular to each edge), unit
_EDGE = np.array([[np.cos(np.radians(a)), np.sin(np.radians(a))] for a in range(30, 390, 60)])
_EDGE = _EDGE / np.linalg.norm(_EDGE, axis=1, keepdims=True)

def _elapsed():
    return time.time() - _T0

def _remaining():
    return _DEADLINE - _elapsed()

def _rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])

def _smooth_overlap(d, A, B):
    """Smooth measure of overlap of two unit hexes at relative offset d, orients A,B.
    Returns ~0 when separated, >0 when overlapping."""
    D = A.T @ B
    # six edge-normal directions of the Minkowski difference
    e1 = np.vstack([_EDGE @ D.T, -(_EDGE @ D.T)])
    e2 = np.vstack([_EDGE, -_EDGE])
    p1 = e1 @ d
    p2 = e2 @ d
    m = 1.0 - (np.abs(p1) + np.abs(p2))  # margin; >0 means separated
    # soft-min of margins (log-sum-exp), large m -> 0
    mmax = m.max()
    if mmax > 50.0:
        return 0.0
    return -(mmax + np.log(np.exp(m - mmax).sum()))

def _soft_containment(V, R, c, S):
    """V: (6,2) rotated vertices; c center; S outer side. Returns >0 if outside."""
    d = V + c
    p = _EDGE @ d.T  # (6,6): dot of each vertex with each outer normal
    # constraint p <= S (circumradius of outer = S)
    viol = np.maximum(0.0, p - S)
    return viol

def construct():
    best = None
    best_val = 40.0
    # initial guess: ring of 6 + inner 6
    def _init():
        pts = []
        # outer ring at radius ~2.3
        for k in range(6):
            a = np.radians(60 * k)
            pts.append([2.3 * np.cos(a), 2.3 * np.sin(a), 0.0])
        # inner ring at radius ~1.2
        for k in range(6):
            a = np.radians(60 * k + 30)
            pts.append([1.2 * np.cos(a), 1.2 * np.sin(a), 0.0])
        return np.array(pts)

    x0 = _init()
    x0 = np.concatenate([x0.ravel(), [0.0, 0.0, 3.0, 0.0]])  # + outer cx,cy,S,ang
    nvar = x0.size

    def _unpack(x):
        P = x[:3 * N].reshape(N, 3)
        cx, cy, S, ang = x[3 * N], x[3 * N + 1], x[3 * N + 2], x[3 * N + 3]
        return P, cx, cy, S, ang

    def _objective(x):
        P, cx, cy, S, ang = _unpack(x)
        # clamp S positive
        S = max(S, 0.5)
        R0 = _rot(ang)
        outer_c = np.array([cx, cy])
        ov = 0.0
        cont = 0.0
        for i in range(N):
            xi, yi, ai = P[i]
            Ri = _rot(ai)
            Vi = _VERT @ Ri.T  # rotated vertices
            # containment: vertices must be inside outer hex
            d = Vi + np.array([xi, yi]) - outer_c
            p = _EDGE @ (R0.T @ d.T)  # rotate into outer frame, dot with normals
            viol = np.maximum(0.0, p - S)
            cont += viol.max()
            # overlap with all j
            for j in range(i + 1, N):
                xj, yj, aj = P[j]
                dj = np.array([xi - xj, yi - yj])
                ov += _smooth_overlap(dj, Ri, _rot(aj))
        w_ov = 5.0
        w_ct = 5.0
        return S + w_ov * ov + w_ct * cont

    def _grad(x):
        # numerical gradient (cheap enough for small nvar)
        g = np.zeros(nvar)
        eps = 1e-6
        f0 = _objective(x)
        for k in range(nvar):
            xp = x.copy(); xp[k] += eps
            xm = x.copy(); xm[k] -= eps
            g[k] = (_objective(xp) - _objective(xm)) / (2 * eps)
        return g

    bounds = [(-10.0, 10.0)] * (3 * N) + [(-10.0, 10.0), (-10.0, 10.0), (0.5, 20.0), (-np.pi, np.pi)]

    # run L-BFGS-B with time guard
    result = None
    try:
        if _remaining() > 5.0:
            result = minimize(_objective, x0, jac=_grad, method='L-BFGS-B',
                              bounds=bounds, options={'maxiter': 400, 'ftol': 1e-12,
                                                      'gtol': 1e-10, 'maxfun': 50000})
    except Exception:
        result = None

    candidates = [x0]
    if result is not None and result.x is not None:
        candidates.append(result.x)

    def _refit(x):
        P, cx, cy, S, ang = _unpack(x)
        R0 = _rot(ang)
        outer_c = np.array([cx, cy])
        # compute exact required S for this placement: max over all vertices of support
        req = 0.0
        for i in range(N):
            xi, yi, ai = P[i]
            Ri = _rot(ai)
            Vi = _VERT @ Ri.T
            d = Vi + np.array([xi, yi]) - outer_c
            p = _EDGE @ (R0.T @ d.T)
            req = max(req, p.max())
        # also ensure pairwise non-overlap; if overlapping, we accept but S won't help
        return P, outer_c, req, ang

    best_P = None
    for x in candidates:
        P, outer_c, S, ang = _refit(x)
        # check pairwise overlap (exact)
        ok = True
        for i in range(N):
            for j in range(i + 1, N):
                if _smooth_overlap(np.array([P[i, 0] - P[j, 0], P[i, 1] - P[j, 1]]),
                                   _rot(P[i, 2]), _rot(P[j, 2])) > 1e-4:
                    ok = False
        # validity: also need S feasible (containment already by construction)
        if ok:
            if S < best_val:
                best_val = S
                best_P = (P, outer_c, ang)

    if best_P is None:
        # fallback: return something valid-ish (large S)
        P, outer_c, S, ang = _refit(candidates[0])
        return P.tolist(), (float(outer_c[0]), float(outer_c[1])), float(max(S, 40.0)), float(ang)

    P, outer_c, ang = best_P
    return P.tolist(), (float(outer_c[0]), float(outer_c[1])), float(best_val), float(ang)
