"""Fresh design: hard-disk expansion + SLSQP polish for packing 50 equal circles."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
HALF = 0.5
DEADLINE = time.time() + 140.0  # explicit wall-clock guard (limit 150 s, safety margin)


def _score(centers):
    """Exact objective: largest common radius for disjoint, in-square circles."""
    centers = np.asarray(centers, dtype=float)
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0
    d = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    r = 0.5 * min(float(np.min(d)), float(np.min(HALF - np.abs(centers))))
    return float(r) if np.isfinite(r) and r > 0 else 0.0


def _expand(centers, r0, iterations, step, alpha):
    """Hard-disk expansion: grow radius, push apart overlapping pairs."""
    centers = np.array(centers, dtype=float)
    r = r0
    for _ in range(iterations):
        moved = False
        for i in range(N):
            for j in range(i + 1, N):
                d = centers[i] - centers[j]
                dist = np.linalg.norm(d)
                if dist < 2.0 * r:
                    if dist < 1e-12:
                        d = np.array([0.01, 0.01])
                        dist = np.linalg.norm(d)
                    push = 0.5 * (2.0 * r - dist) * (d / dist)
                    centers[i] += step * push
                    centers[j] -= step * push
                    moved = True
        if not moved:
            r = min(r + step, alpha)
        else:
            # nudge radius up slightly to keep seeking larger packings
            r = min(r + 0.5 * step, alpha)
    # final clamp into the square
    centers = np.clip(centers, -HALF, HALF)
    return centers, r


def _polish(centers, r0):
    """SLSQP polish: maximize r with exact hard constraints."""
    n = N * 2
    idx = np.arange(n)
    iu = np.triu_indices(N, 1)
    dist_idx = idx[iu[0]] * 2 + idx[iu[1]]
    bx = np.array([HALF, -HALF] * N)
    by = np.array([HALF, -HALF] * N)
    r0 = max(r0, 1e-6)

    def obj(v):
        return -v[n]

    def jac(v):
        g = np.zeros(n + 1)
        g[n] = -1.0
        return g

    def con_dist(v):
        c = v[:n].reshape(N, 2)
        d = np.linalg.norm(c[iu[0]] - c[iu[1]], axis=1)
        return d - 2.0 * v[n]

    def con_bx(v):
        c = v[:n].reshape(N, 2)
        return np.concatenate([HALF - c[:, 0], HALF + c[:, 0]])

    def con_by(v):
        c = v[:n].reshape(N, 2)
        return np.concatenate([HALF - c[:, 1], HALF + c[:, 1]])

    x0 = np.concatenate([centers.ravel(), [r0]])
    cons = [
        {'type': 'ineq', 'fun': con_dist},
        {'type': 'ineq', 'fun': con_bx},
        {'type': 'ineq', 'fun': con_by},
    ]
    try:
        res = minimize(obj, x0, method='SLSQP', jac=jac, constraints=cons,
                       bounds=[(bx[i], by[i]) for i in range(n)] + [(0.0, HALF)],
                       options={'maxiter': 150, 'ftol': 1e-12})
        if res.success or res.x is not None:
            c = res.x[:n].reshape(N, 2)
            c = np.clip(c, -HALF, HALF)
            return c
    except Exception:
        pass
    return centers


def _init_grid():
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _init_rows(rows, cols):
    pts = []
    for i in range(rows):
        y = (i + 0.5) / rows - 0.5
        for j in range(cols):
            x = (j + 0.5) / cols - 0.5
            pts.append((x, y))
    return np.array(pts[:N], dtype=float)


def _init_honeycomb():
    # offset (brick) lattice: denser than a plain square grid
    pts = []
    rowh = 0.8660254
    row = 0
    while len(pts) < N:
        y = (row + 0.5) / (rowh * (N // 7 + 1)) - 0.5
        row = row + 1
        if row > 12:
            break
        xoff = 0.5 if (row % 2) else 0.0
        cols = 7
        for j in range(cols):
            x = (j + 0.5) / cols - 0.5 + xoff / cols
            pts.append((x, y))
    return np.array(pts[:N], dtype=float)


def construct():
    t0 = time.time()
    candidates = []

    # multi-start initializations
    starts = [_init_grid(), _init_rows(5, 10), _init_honeycomb(), _init_rows(10, 5)]

    for s in starts:
        if time.time() > DEADLINE:
            break
        # hard-disk expansion to a good radius
        c_exp, r_exp = _expand(s, 0.05, 600, 0.006, 0.075)
        candidates.append(c_exp)
        # SLSQP polish from the expanded configuration
        if time.time() > DEADLINE:
            break
        c_pol = _polish(c_exp, r_exp)
        candidates.append(c_pol)
        # also polish the raw start (different basin)
        if time.time() > DEADLINE:
            break
        c_pol2 = _polish(s, 0.05)
        candidates.append(c_pol2)

    # pick the best by exact score
    best = None
    best_score = -1.0
    for c in candidates:
        sc = _score(c)
        if sc > best_score:
            best_score = sc
            best = c

    if best is None:
        best = _init_grid()
    # final safety clamp
    best = np.clip(np.asarray(best, dtype=float), -HALF, HALF)
    return best
