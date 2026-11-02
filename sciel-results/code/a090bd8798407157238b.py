"""Structural change: position-based dynamics (expand-and-relax) for circle packing."""
import time
import numpy as np

N = 32

_DEADLINE = 150.0
_TOL = 1e-7


def _valid_and_score(arr):
    """Return sum of radii if arr is a valid packing, else -inf."""
    x = arr[:, 0]
    y = arr[:, 1]
    r = arr[:, 2]
    if np.any(r < 0) or np.any(np.isnan(r)):
        return -np.inf
    # boundary
    if np.any(x - r < -_TOL) or np.any(x + r > 1.0 + _TOL):
        return -np.inf
    if np.any(y - r < -_TOL) or np.any(y + r > 1.0 + _TOL):
        return -np.inf
    # pairwise
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            if d < r[i] + r[j] - _TOL:
                return -np.inf
    return float(np.sum(r))


def _hex_init(scale=1.0, jitter=0.0):
    """Hexagonal lattice of N circles, scaled into the unit square with margin."""
    # hex packing: rows offset; spacing = 2*1 (diameter unit) in lattice units
    # We'll generate a hex lattice of points with unit spacing then scale to fit.
    pts = []
    row = 0
    while len(pts) < N:
        for col in range(6):
            x = col + (0.5 if row % 2 else 0.0)
            y = row * np.sqrt(3.0) / 2.0
            pts.append((x, y))
            if len(pts) >= N:
                break
        row += 1
    pts = np.array(pts[:N])
    # normalize to fit in [0,1] with a small margin
    mx, my = pts.max(axis=0)
    mn, mn_y = pts.min(axis=0), pts.min(axis=1)
    pts = (pts - mn) / max(mx - mn, my - mn_y)  # scale to [0,1]
    pts = pts * scale + (1.0 - scale) / 2.0
    if jitter > 0:
        pts = pts + jitter * (np.random.rand(N, 2) - 0.5)
    r0 = 0.03  # small initial radius
    return np.column_stack([pts, np.full(N, r0)])


def _relax(arr, steps=400, grow_per_step=0.002):
    """Expand radii and project out overlaps and boundary penetrations."""
    arr = arr.copy()
    x = arr[:, 0].copy()
    y = arr[:, 1].copy()
    r = arr[:, 2].copy()
    for _ in range(steps):
        # grow
        r += grow_per_step
        # constraint projection iterations
        for _it in range(8):
            # boundary
            for i in range(N):
                if x[i] - r[i] < 0.0:
                    x[i] = r[i]
                if x[i] + r[i] > 1.0:
                    x[i] = 1.0 - r[i]
                if y[i] - r[i] < 0.0:
                    y[i] = r[i]
                if y[i] + r[i] > 1.0:
                    y[i] = 1.0 - r[i]
            # pairwise: push apart
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[j] - x[i]
                    dy = y[j] - y[i]
                    d = np.hypot(dx, dy)
                    min_d = r[i] + r[j]
                    if d < min_d:
                        if d < 1e-9:
                            dx, dy, d = 1e-9, 0.0, 1e-9
                        # push each half the overlap (equal mass)
                        push = (min_d - d) * 0.5
                        nx, ny = dx / d, dy / d
                        x[i] -= push * nx
                        y[i] -= push * ny
                        x[j] += push * nx
                        y[j] += push * ny
        # after projection, re-clamp to boundary (projection may push out)
        for i in range(N):
            x[i] = min(max(x[i], r[i]), 1.0 - r[i])
            y[i] = min(max(y[i], r[i]), 1.0 - r[i])
    arr[:, 0] = x
    arr[:, 1] = y
    arr[:, 2] = r
    return arr


def construct():
    start = time.time()
    best_arr = None
    best_score = -np.inf

    seeds = [1, 7, 42, 123, 999, 2024, 31337, 555]
    for si, seed in enumerate(seeds):
        if time.time() > _DEADLINE - 5:
            break
        np.random.seed(seed)
        for scale in [0.9, 1.0]:
            for jitter in [0.0, 0.01]:
                arr0 = _hex_init(scale=scale, jitter=jitter)
                # progressive relaxation: several passes with decreasing growth
                arr = arr0.copy()
                for phase in range(3):
                    steps = 150
                    grow = [0.004, 0.002, 0.0008][phase]
                    arr = _relax(arr, steps=steps, grow_per_step=grow)
                    sc = _valid_and_score(arr)
                    if sc > best_score:
                        best_score = sc
                        best_arr = arr.copy()
                if time.time() > _DEADLINE - 5:
                    break
            if time.time() > _DEADLINE - 5:
                break
        if time.time() > _DEADLINE - 5:
            break

    if best_arr is None:
        # fallback: seed grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # final safety: if best is invalid for some reason, fall back
    if _valid_and_score(best_arr) < 0:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    return best_arr
