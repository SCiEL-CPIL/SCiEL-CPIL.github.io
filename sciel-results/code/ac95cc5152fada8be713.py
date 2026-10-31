"""Circle packing in perimeter-4 rectangle: gradient-based constrained optimization."""
import time
import numpy as np
from scipy.optimize import minimize

N = 21
_EPS = 1e-4


def _feasible(circles):
    """Check all constraints; return True if valid."""
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    w = float(np.max(x) + np.max(r))
    h = float(np.max(y) + np.max(r))
    if 2.0 * (w + h) > 4.0 + 1e-7:
        return False
    # pairwise
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    need = r[:, None] + r[None, :] + _EPS
    # ignore diagonal
    idx = np.triu_indices(N, 1)
    if np.any(d[idx] < need[idx]):
        return False
    return True


def _build_constraints(w, h):
    def f_eq(v):
        return [v[2 * N] + v[2 * N + 1] - (w + h)]

    def f_ineq(v):
        x = v[0:N]
        y = v[N:2 * N]
        r = v[2 * N:3 * N]
        c = []
        c.append(w - (np.max(x) + np.max(r)))
        c.append(h - (np.max(y) + np.max(r)))
        c.append(1.0 - (2.0 * (w + h)))
        c.append(w - h)  # w <= h
        c.append(1e-3 - np.min(r))
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx * dx + dy * dy)
        need = r[:, None] + r[None, :] + _EPS
        iu = np.triu_indices(N, 1)
        c.extend((d[iu] - need[iu]).tolist())
        return np.array(c)

    return {'eq': f_eq, 'ineq': f_ineq}


def _init_grid(w, h):
    side = int(np.ceil(np.sqrt(N)))
    xs = []
    ys = []
    rs = []
    for i in range(side):
        for j in range(side):
            xs.append((i + 0.5) * (w / side))
            ys.append((j + 0.5) * (h / side))
            rs.append(0.5 * (w / side))
    # scale radii so they fit within bounds with clearance
    maxr = 0.5 * min(w, h)
    scale = min(1.0, maxr / max(rs))
    rs = [r * scale for r in rs]
    x = np.array(xs[:N])
    y = np.array(ys[:N])
    r = np.array(rs[:N])
    # clip into bounds
    r = np.minimum(r, np.minimum(w - x, x))
    r = np.minimum(r, np.minimum(h - y, y))
    r = np.maximum(r, 1e-3)
    return np.concatenate([x, y, r])


def _init_random(w, h, rng):
    x = rng.uniform(0, w, N)
    y = rng.uniform(0, h, N)
    r = rng.uniform(0.02, 0.12, N)
    r = np.minimum(r, np.minimum(w - x, x))
    r = np.minimum(r, np.minimum(h - y, y))
    r = np.maximum(r, 1e-3)
    return np.concatenate([x, y, r])


def _score(circles):
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    w = float(np.max(x) + np.max(r))
    h = float(np.max(y) + np.max(r))
    if 2.0 * (w + h) > 4.0 + 1e-7:
        return -1e9
    if not _feasible(circles):
        return -1e9
    return float(np.sum(r))


def construct():
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin under 120s
    rng = np.random.default_rng(0)

    best = None
    best_score = -1e9

    def consider(circles):
        nonlocal best, best_score
        if _feasible(circles):
            s = _score(circles)
            if s > best_score:
                best_score = s
                best = circles.copy()

    # Candidate rectangles (w<=h, 2(w+h)<=4)
    rects = [(1.0, 1.0), (0.8, 1.2), (0.6, 1.4), (1.2, 0.8), (1.4, 0.6), (0.5, 1.5)]
    rects = [(w, h) for (w, h) in rects if 2.0 * (w + h) <= 4.0 + 1e-9]
    if not rects:
        rects = [(1.0, 1.0)]

    # Always seed with the square grid first
    for w, h in rects:
        consider(_init_grid(w, h))

    # Optimization restarts: grid + random starts, annealed step sizes
    step_sizes = [1.0, 0.6, 0.3, 0.15, 0.07, 0.03]

    for w, h in rects:
        cons = _build_constraints(w, h)
        starts = [_init_grid(w, h)]
        starts += [_init_random(w, h, rng) for _ in range(4)]
        for ss in step_sizes:
            for st in starts:
                if time.time() > deadline:
                    break
                try:
                    res = minimize(
                        lambda v: -np.sum(v[2 * N:3 * N]),
                        st,
                        method='SLSQP',
                        constraints=cons,
                        options={'maxiter': 200, 'ftol': 1e-8,
                                 'eps': 1e-6 * max(1.0, ss)},
                    )
                except Exception:
                    continue
                if res.x is not None and np.all(np.isfinite(res.x)):
                    circles = np.array([res.x[0:N], res.x[N:2 * N], res.x[2 * N:3 * N]]).T
                    consider(circles)
            if time.time() > deadline:
                break

    if best is None:
        # fallback: square grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return best


if __name__ == "__main__":
    c = construct()
    print("n circles:", c.shape[0])
    print("sum radii:", _score(c))
    print("perimeter:", 2 * ((np.max(c[:, 0]) + np.max(c[:, 2])) + (np.max(c[:, 1]) + np.max(c[:, 2]))))
