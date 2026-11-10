"""Packing 50 equal circles in the unit disk via simulated annealing."""
import numpy as np
import time

N = 50
DEADLINE = 150.0
SAFETY = 10.0          # stop this many seconds before the wall-clock limit
RNG_SEED = 12345


def _score(pts):
    """Largest common radius for the given centres."""
    if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
        return 0.0
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    min_pair = d.min()
    min_bound = 1.0 - np.linalg.norm(pts, axis=1).max()
    return min(min_pair / 2.0, min_bound)


def _hex_start():
    """Dense hexagonal lattice clipped to the disk, then trimmed to N."""
    r0 = 0.1258254895304
    spacing = 2.0 * r0
    s3 = np.sqrt(3.0)
    k = 5
    pts = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * s3 / 2.0
            if np.hypot(x, y) + r0 <= 1.0:
                pts.append((x, y))
    pts = np.array(pts)
    if len(pts) >= N:
        order = np.argsort(np.hypot(pts[:, 0], pts[:, 1]), kind="stable")
        return pts[order[:N]]
    # fallback: uniform random
    ang = np.random.uniform(0, 2 * np.pi, N)
    rad = np.random.uniform(0, 0.95, N)
    return np.column_stack([rad * np.cos(ang), rad * np.sin(ang)])


def construct():
    t0 = time.time()
    np.random.seed(RNG_SEED)

    pts = _hex_start().astype(float)
    best = pts.copy()
    best_score = _score(pts)

    cur_score = best_score
    cur = pts.copy()

    # Annealing schedule
    T0 = 0.02
    Tend = 1e-4
    # number of iterations to use up the budget
    budget = max(0.0, DEADLINE - SAFETY - (time.time() - t0))
    # estimate per-iteration cost to pick a reasonable N_iter
    N_iter = 2_000_000
    # temperature geometric schedule
    ratio = (Tend / T0) ** (1.0 / max(1, N_iter))

    T = T0
    # step size (in units of radius)
    step = 0.05

    it = 0
    while it < N_iter and (time.time() - t0) < (DEADLINE - SAFETY):
        # pick a random circle and nudge it
        i = np.random.randint(N)
        ang = np.random.uniform(0, 2 * np.pi)
        dx = np.random.normal(0, step)
        dy = np.random.normal(0, step)
        newx = cur[i, 0] + dx
        newy = cur[i, 1] + dy

        # keep inside container with margin
        if np.hypot(newx, newy) + cur_score > 1.0:
            # reflect toward origin
            r = np.hypot(newx, newy)
            if r > 0:
                scale = (1.0 - cur_score) / r
                newx *= scale
                newy *= scale

        old = cur[i, 0], cur[i, 1]
        cur[i, 0] = newx
        cur[i, 1] = newy
        s = _score(cur)

        if s >= cur_score or np.random.random() < np.exp((s - cur_score) / T):
            cur_score = s
            if s > best_score:
                best_score = s
                best = cur.copy()
        else:
            cur[i, 0], cur[i, 1] = old

        it += 1
        T *= ratio

    # final repair: project any circle that violates the container
    best = _repair(best, best_score)
    return best


def _repair(pts, r):
    """Nudge circles back inside the disk while preserving radius r."""
    pts = pts.copy()
    for i in range(N):
        rad = np.hypot(pts[i, 0], pts[i, 1])
        if rad > 1.0 - r:
            scale = (1.0 - r) / rad if rad > 0 else 1.0
            pts[i, 0] *= scale
            pts[i, 1] *= scale
    return pts
