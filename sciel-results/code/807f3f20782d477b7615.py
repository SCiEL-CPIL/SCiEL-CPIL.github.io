"""Hexagonal seed + vectorized annealing for 50 equal circles in the unit square."""
import numpy as np
import time

N = 50
HALF = 0.5
T_START = 0.05
T_END = 1e-9


def _score(c):
    """Largest common radius for centres c (N,2)."""
    c = np.asarray(c, dtype=np.float64)
    if c.shape != (N, 2) or not np.all(np.isfinite(c)):
        return 0.0
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    min_bound = np.min(HALF - np.abs(c))
    return min(min_pair, min_bound)


def _hex_seed():
    """Hexagonal lattice seed inside the square, clipped to bounds."""
    r = 0.0713
    dx = 2.0 * r
    dy = np.sqrt(3.0) * r
    xs, ys = [], []
    row = 0
    y = -HALF + r
    while y <= HALF - r + 1e-9:
        x = -HALF + r + (0.5 * dx if (row % 2) else 0.0)
        while x <= HALF - r + 1e-9:
            xs.append(x)
            ys.append(y)
            x += dx
        y += dy
        row += 1
    pts = np.array(list(zip(xs, ys)))
    # keep the N points closest to the centre
    if len(pts) >= N:
        d2 = np.sum(pts * pts, axis=1)
        idx = np.argsort(d2)[:N]
        pts = pts[idx]
    else:
        # fallback: top up with a grid if lattice too small
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, yy) for yy in ticks for x in ticks])
        pts = np.vstack([pts, grid[: N - len(pts)]])
    return np.clip(pts, -HALF + r, HALF - r)


def construct():
    deadline = time.time() + 140.0   # safety margin before 150 s kill
    pts = _hex_seed().copy()
    best = pts.copy()
    best_score = _score(best)

    n_steps = 40000
    for it in range(n_steps):
        if time.time() > deadline:
            break
        T = T_START * (T_END / T_START) ** (it / max(1, n_steps - 1))
        # random perturbation scaled by temperature
        step = 0.004 + 0.02 * T
        cand = pts + np.random.randn(N, 2) * step
        cand = np.clip(cand, -HALF, HALF)

        d = cand[:, None, :] - cand[None, :, :]
        dist = np.sqrt(np.sum(d * d, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_pair = np.min(dist) / 2.0
        min_bound = np.min(HALF - np.abs(cand))
        s = min(min_pair, min_bound)

        if s > best_score:
            best_score = s
            best = cand.copy()
        else:
            # Metropolis accept to escape local minima
            if np.random.rand() < np.exp((s - best_score) / max(T, 1e-12)):
                pts = cand
            else:
                pts = cand  # keep exploring; best is preserved separately

    return best
