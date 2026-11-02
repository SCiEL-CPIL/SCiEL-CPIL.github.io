import time
import numpy as np

N = 21
DEADLINE = 150.0
MARGIN = 3.0  # safety margin in seconds

def _validity(x, r, W, H, eps=1e-9):
    # boundaries
    if np.any(x[0] - r < -eps) or np.any(x[0] + r > W + eps):
        return False
    if np.any(x[1] - r < -eps) or np.any(x[1] + r > H + eps):
        return False
    # pairwise disjoint interiors
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i, 0] - x[j, 0]
            dy = x[i, 1] - x[j, 1]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - 1e-9:
                return False
    return True

def _solve_aspect(a, rng, t_start, deadline):
    # perimeter = 2*(W+H) = 4  =>  W+H = 2
    H = 2.0 / (1.0 + a)
    W = a * H
    # radius upper bound: each circle must fit, and 21 circles total area <= W*H
    # r <= min(W,H)/2 and sum of areas <= W*H
    r_max_area = np.sqrt(W * H / (np.pi * N))
    r_bound = min(W, H) * 0.5
    r_init = min(0.5 * min(W, H), r_max_area) * 0.9

    best_val = -1.0
    best_x = None
    best_r = None

    n_restarts = 0
    while True:
        if time.time() - t_start > deadline - MARGIN:
            break
        # random restart
        x = rng.uniform(r_init, W - r_init, size=(N, 2))
        r = np.full(N, r_init)
        # small random radii variation
        r = r * rng.uniform(0.85, 1.0, size=N)

        # annealing schedule
        steps = 3000
        T = 0.05
        T_min = 1e-4
        decay = (T_min / T) ** (1.0 / steps)
        repel_k = 2.0
        attract_k = 0.05
        wall_k = 5.0

        for s in range(steps):
            if time.time() - t_start > deadline - MARGIN:
                break
            T = max(T * decay, T_min)
            # compute forces
            fx = np.zeros((N, 2))
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i, 0] - x[j, 0]
                    dy = x[i, 1] - x[j, 1]
                    d2 = dx * dx + dy * dy
                    d = np.sqrt(d2) if d2 > 1e-12 else 1e-6
                    rs = r[i] + r[j]
                    if d < rs:
                        # repulsion
                        f = repel_k * (rs - d)
                        fx[i, 0] += f * dx / d
                        fx[i, 1] += f * dy / d
                        fx[j, 0] -= f * dx / d
                        fx[j, 1] -= f * dy / d
                    else:
                        # mild attraction to encourage touching (helps fill)
                        f = -attract_k * (d - rs) * 0.05
                        fx[i, 0] += f * dx / d
                        fx[i, 1] += f * dy / d
                        fx[j, 0] -= f * dx / d
                        fx[j, 1] -= f * dy / d
                # walls
                # left
                if x[i, 0] < r[i]:
                    fx[i, 0] += wall_k * (r[i] - x[i, 0])
                # right
                if x[i, 0] > W - r[i]:
                    fx[i, 0] -= wall_k * (x[i, 0] - (W - r[i]))
                # bottom
                if x[i, 1] < r[i]:
                    fx[i, 1] += wall_k * (r[i] - x[i, 1])
                # top
                if x[i, 1] > H - r[i]:
                    fx[i, 1] -= wall_k * (x[i, 1] - (H - r[i]))

            # update
            step = T
            x += step * fx
            # clip to box (soft, keep inside)
            x[:, 0] = np.clip(x[:, 0], r, W - r)
            x[:, 1] = np.clip(x[:, 1], r, H - r)
            # try to grow radii a little each step (uniform scaling)
            # increase all radii slightly, keep proportional
            r *= (1.0 + 1e-5)
            # enforce radius upper bound
            r = np.minimum(r, r_bound)

        # after annealing, try to maximize radii: scale all up until invalid
        # binary search on scale factor
        lo, hi = 1.0, 2.0
        for _ in range(40):
            mid = (lo + hi) / 2
            if _validity(x, r * mid, W, H):
                lo = mid
            else:
                hi = mid
        scale = lo * (1 - 1e-4)
        r_final = r * scale
        val = np.sum(r_final)
        if val > best_val and _validity(x, r_final, W, H):
            best_val = val
            best_x = x.copy()
            best_r = r_final.copy()
        n_restarts += 1

    if best_x is None:
        return None
    # convert to (x, y, r) in absolute coords; bounding rect is W x H
    return np.column_stack([best_x, best_r]), best_val, W, H

def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE
    rng = np.random.default_rng(2024)

    # aspect ratio grid (a = W/H). Perimeter fixes scale, so only shape matters.
    # Search a range of aspect ratios around 1.
    aspects = np.concatenate([
        np.logspace(np.log10(0.5), np.log10(2.0), 25),
    ])
    # sort by closeness to 1 to get good values early
    aspects = aspects[np.argsort(np.abs(np.log(aspects)))]

    best_overall = None
    best_val_overall = -1.0

    # per-aspect time budget
    n_aspects = len(aspects)
    per_aspect_time = (DEADLINE - MARGIN - 2.0) / n_aspects

    for a in aspects:
        if time.time() - t_start > deadline - MARGIN:
            break
        a_t0 = time.time()
        a_deadline = min(a_t0 + per_aspect_time * 1.5, deadline - MARGIN)
        res = _solve_aspect(a, rng, t_start, a_deadline)
        if res is not None:
            arr, val, W, H = res
            if val > best_val_overall:
                best_val_overall = val
                best_overall = arr

    if best_overall is None:
        # fallback: simple grid (always valid-ish)
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return best_overall
