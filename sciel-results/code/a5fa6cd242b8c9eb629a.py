"""Circle packing in unit square: optimize centers, radii = min dist to boundary/neighbors."""
import time
import numpy as np
from scipy.spatial import Delaunay, cKDTree

N = 26
DEADLINE = 110.0  # safety margin under 120s/150s
T0 = time.time()


def _time_left():
    return DEADLINE - (time.time() - T0)


def _compute_radii(c):
    """c: (N,2). Return radii = min(dist to boundary, half min neighbor dist)."""
    x, y = c[:, 0], c[:, 1]
    r = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    if N >= 2:
        tree = cKDTree(c)
        # nearest neighbor distance
        d, _ = tree.query(c, k=2)  # k=2 -> first is self
        d = d[:, 1]
        r = np.minimum(r, 0.5 * d)
    return r


def _objective(c, temp):
    """Smooth surrogate: sum of soft-min radii. Higher = better."""
    x, y = c[:, 0], c[:, 1]
    # boundary distances
    b = np.stack([x, 1.0 - x, y, 1.0 - y], axis=1)  # (N,4)
    # neighbor distances
    if N >= 2:
        tree = cKDTree(c)
        d, _ = tree.query(c, k=2)
        d = d[:, 1]  # (N,)
        # candidate distances for each circle: 4 boundary + 1 neighbor (halved)
        cand = np.concatenate([b, 0.5 * d[:, None]], axis=1)  # (N,5)
    else:
        cand = b
    # soft-min via log-sum-exp: min approx = -logsumexp(-cand/temp)*temp
    neg = -cand / temp
    m = neg.max(axis=1, keepdims=True)
    lse = m + np.log(np.exp(neg - m).sum(axis=1, keepdims=True))
    softmin = -lse * temp  # (N,1)
    return float(softmin.sum())


def _gradient(c, temp):
    """Gradient of the objective w.r.t. centers (numerical, vectorized-ish)."""
    grad = np.zeros_like(c)
    eps = 1e-6
    f0 = _objective(c, temp)
    for i in range(N):
        for j in range(2):
            cp = c.copy()
            cp[i, j] += eps
            fp = _objective(cp, temp)
            cm = c.copy()
            cm[i, j] -= eps
            fm = _objective(cm, temp)
            grad[i, j] = (fp - fm) / (2 * eps)
    return grad


def _local_optimize(c, temp, iters, step0):
    """Gradient ascent on the smooth surrogate with adaptive step."""
    c = c.copy()
    f = _objective(c, temp)
    step = step0
    for it in range(iters):
        if _time_left() < 5:
            break
        g = _gradient(c, temp)
        gn = np.linalg.norm(g)
        if gn < 1e-12:
            break
        g = g / gn
        # try step
        cand = c + step * g
        cand = np.clip(cand, 1e-3, 1.0 - 1e-3)
        fnew = _objective(cand, temp)
        if fnew > f:
            c = cand
            f = fnew
            step = min(step * 1.05, 0.05)
        else:
            step *= 0.5
            if step < 1e-5:
                break
    return c, f


def _physics_relax(c, iters):
    """Push overlapping circles apart, then grow radii. Returns centers."""
    c = c.copy()
    for it in range(iters):
        if _time_left() < 5:
            break
        r = _compute_radii(c)
        # forces
        f = np.zeros_like(c)
        for i in range(N):
            for j in range(i + 1, N):
                dvec = c[i] - c[j]
                dist = np.linalg.norm(dvec)
                minr = r[i] + r[j]
                if dist < minr and dist > 1e-9:
                    push = (minr - dist) / dist
                    f[i] += push * dvec
                    f[j] -= push * dvec
        # wall forces
        for i in range(N):
            for axis in range(2):
                if c[i, axis] < r[i]:
                    f[i, axis] += (r[i] - c[i, axis])
                if c[i, axis] > 1.0 - r[i]:
                    f[i, axis] -= (c[i, axis] - (1.0 - r[i]))
        c = c + 0.1 * f
        c = np.clip(c, 0.0, 1.0)
    return c


def construct():
    # ---- initial layout: grid ----
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side) for i in range(side) for j in range(side)]
    c0 = np.array(cells[:N], dtype=float)

    best_c = c0.copy()
    best_val = -np.inf
    r = _compute_radii(c0)
    best_val = float(r.sum())

    # ---- multi-phase smooth optimization with annealing temperature ----
    temps = [0.05, 0.02, 0.005, 0.001]
    for temp in temps:
        if _time_left() < 8:
            break
        # perturb + local optimize
        for restart in range(3):
            if _time_left() < 8:
                break
            c = best_c.copy()
            c = c + np.random.randn(N, 2) * 0.02
            c = np.clip(c, 1e-3, 1.0 - 1e-3)
            c, f = _local_optimize(c, temp, iters=40, step0=0.02)
            if f > best_val:
                best_val = f
                best_c = c.copy()
        # physics relaxation
        if _time_left() < 8:
            break
        c = _physics_relax(best_c, iters=60)
        r = _compute_radii(c)
        if r.sum() > best_val:
            best_val = float(r.sum())
            best_c = c.copy()

    # ---- final high-precision polish: tighten centers with small steps ----
    if _time_left() > 6:
        c = best_c.copy()
        for it in range(200):
            if _time_left() < 3:
                break
            # gradient ascent with tiny step on exact objective
            g = _gradient(c, 0.001)
            gn = np.linalg.norm(g)
            if gn < 1e-10:
                break
            g = g / gn
            for step in [0.005, 0.002, 0.001, 0.0005]:
                cand = np.clip(c + step * g, 1e-4, 1.0 - 1e-4)
                rc = _compute_radii(cand)
                if rc.sum() > _compute_radii(c).sum() + 1e-9:
                    c = cand
                    break
        r = _compute_radii(c)
        if r.sum() > best_val:
            best_val = float(r.sum())
            best_c = c.copy()

    # ---- build output ----
    c = best_c
    r = _compute_radii(c)
    # ensure strict non-overlap and inside: shrink slightly
    r = r * (1.0 - 1e-4)
    out = np.column_stack([c, r])
    return out
