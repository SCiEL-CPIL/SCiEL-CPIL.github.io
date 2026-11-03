"""Circle packing in unit square: maximize sum of radii of 26 disjoint circles."""
import numpy as np

N = 26


def _feasible(x, y, r, eps=1e-9):
    # walls
    if np.any(r < -eps):
        return False
    if np.any(x - r < -eps) or np.any(x + r > 1 + eps):
        return False
    if np.any(y - r < -eps) or np.any(y + r > 1 + eps):
        return False
    # overlaps
    d = np.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2)
    need = r[:, None] + r[None, :]
    if np.any(d < need - eps):
        return False
    return True


def _objective(x, y, r):
    return float(np.sum(r))


def _grow_separate(x, y, r, iters, grow_rate=0.02, sep_rate=0.5, rng=None):
    n = len(x)
    x = x.copy(); y = y.copy(); r = r.copy()
    for it in range(iters):
        # 1) Try to grow each radius limited by walls and neighbors
        for i in range(n):
            # wall-limited radius
            wall_r = min(x[i], 1 - x[i], y[i], 1 - y[i])
            # neighbor-limited
            for j in range(n):
                if i == j:
                    continue
                d = np.hypot(x[i] - x[j], y[i] - y[j])
                if d > 1e-12:
                    lim = (d - r[j])
                    # we can grow r_i until r_i = d - r_j
                    wall_r = min(wall_r, lim)
            r[i] = max(0.0, min(r[i] + grow_rate, wall_r - 1e-9))
        # 2) Separate overlaps (push apart, shrink if needed)
        for _ in range(3):
            moved = False
            for i in range(n):
                for j in range(i + 1, n):
                    dx = x[j] - x[i]; dy = y[j] - y[i]
                    d = np.hypot(dx, dy)
                    need = r[i] + r[j]
                    if d < need - 1e-12:
                        if d < 1e-12:
                            dx = 1e-6; dy = 0.0; d = 1e-6
                        # push apart along direction
                        push = (need - d) / 2.0 * sep_rate
                        ux = dx / d; uy = dy / d
                        x[i] -= push * ux; y[i] -= push * uy
                        x[j] += push * ux; y[j] += push * uy
                        moved = True
            if not moved:
                break
        # 3) Clamp to walls, shrink radius if it pokes out
        for i in range(n):
            x[i] = min(max(x[i], r[i]), 1 - r[i])
            y[i] = min(max(y[i], r[i]), 1 - r[i])
    return x, y, r


def construct():
    import time
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin under 120s

    rng = np.random.default_rng(12345)

    # ---- Initial feasible seed: grid of equal circles ----
    side = int(np.ceil(np.sqrt(N)))  # 5
    cells = [((i + 0.5) / side, (j + 0.5) / side) for i in range(side) for j in range(side)]
    x = np.array([c[0] for c in cells[:N]])
    y = np.array([c[1] for c in cells[:N]])
    r = np.full(N, 0.5 / side - 1e-3)

    best = _objective(x, y, r)
    bx, by, br = x.copy(), y.copy(), r.copy()

    # ---- Multi-restart optimization ----
    iters_per_round = 400
    grow_rates = [0.05, 0.03, 0.02, 0.01, 0.005]

    # First: pure grow/separate from grid
    x, y, r = _grow_separate(x, y, r, iters=iters_per_round, grow_rate=0.03)
    val = _objective(x, y, r)
    if val > best and _feasible(x, y, r):
        best = val; bx, by, br = x.copy(), y.copy(), r.copy()

    while time.time() < deadline:
        # Random perturbation of positions (keep radii from best) to escape local optima
        x0 = bx + rng.uniform(-0.02, 0.02, N)
        y0 = by + rng.uniform(-0.02, 0.02, N)
        r0 = br.copy()
        # clamp
        x0 = np.clip(x0, r0, 1 - r0); y0 = np.clip(y0, r0, 1 - r0)

        # pick a grow rate (coarse to fine)
        gr = grow_rates[int(rng.integers(0, len(grow_rates)))]
        x, y, r = _grow_separate(x0, y0, r0, iters=iters_per_round, grow_rate=gr)
        if _feasible(x, y, r):
            val = _objective(x, y, r)
            if val > best:
                best = val; bx, by, br = x.copy(), y.copy(), r.copy()

    # Final safety: ensure feasibility of returned best
    if not _feasible(bx, by, br):
        # fallback to grid
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    return np.column_stack((bx, by, br))
