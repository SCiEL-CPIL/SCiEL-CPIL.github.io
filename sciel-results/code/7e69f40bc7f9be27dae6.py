"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles."""
import time
import numpy as np

N = 26
START = time.time()
DEADLINE = 115.0  # safety margin before 120s kill (and 150s wall-clock)


def _validity(x, y, r, eps=1e-9):
    """Return True if all circles are disjoint and inside [0,1]^2."""
    if np.any(r < -eps) or np.any(x < -eps) or np.any(y < -eps) or np.any(x > 1 + eps) or np.any(y > 1 + eps):
        return False
    # pairwise: distance >= r_i + r_j
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            if d < r[i] + r[j] - 1e-7:
                return False
    return True


def _sum_r(x, y, r):
    return float(np.sum(r))


def _initial_packing():
    """Grid of equal circles, take first N."""
    side = int(np.ceil(np.sqrt(N)))  # 5
    cells = []
    for j in range(side):
        for i in range(side):
            cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4))
    return np.array(cells[:N])


def _random_packing(rng, base_r=0.11):
    """Random feasible-ish packing with small circles."""
    x = rng.uniform(0.15, 0.85, N)
    y = rng.uniform(0.15, 0.85, N)
    r = np.full(N, base_r)
    return x, y, r


def _make_valid(x, y, r):
    """Shrink radii / nudge positions to get a valid packing (best effort)."""
    for _ in range(2000):
        if _validity(x, y, r):
            return x, y, r
        # find the most violated pair
        worst = 0
        worst_i, worst_j = 0, 1
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(x[i] - x[j], y[i] - y[j])
                viol = (r[i] + r[j]) - d
                if viol > worst:
                    worst = viol
                    worst_i, worst_j = i, j
        # shrink both a bit
        r[worst_i] = max(1e-4, r[worst_i] * 0.98)
        r[worst_j] = max(1e-4, r[worst_j] * 0.98)
        # also check boundaries
        for k in range(N):
            if x[k] < r[k]:
                x[k] = r[k]
            if x[k] > 1 - r[k]:
                x[k] = 1 - r[k]
            if y[k] < r[k]:
                y[k] = r[k]
            if y[k] > 1 - r[k]:
                y[k] = 1 - r[k]
    return x, y, r


def _anneal(x, y, r, rng, steps, t0, t1):
    """Simulated annealing over positions and radii. Returns best (x,y,r,sum)."""
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    best_val = _sum_r(x, y, r)
    cur_val = best_val
    n_steps = max(1, int(steps))
    for s in range(n_steps):
        if time.time() > DEADLINE:
            break
        frac = s / n_steps
        T = t0 * (t1 / t0) ** frac
        # pick a circle
        k = rng.integers(N)
        # propose move: random position + radius perturbation
        dx = rng.normal(0, 0.03)
        dy = rng.normal(0, 0.03)
        dr = rng.normal(0, 0.015)
        nx = x[k] + dx
        ny = y[k] + dy
        nr = r[k] + dr
        # clamp to box
        nx = min(1 - nr, max(nr, nx)) if nr > 0 else x[k]
        ny = min(1 - nr, max(nr, ny)) if nr > 0 else y[k]
        nr = max(1e-4, min(0.5, nr))
        if nr <= 0:
            continue
        # quick overlap check with others
        ok = True
        for j in range(N):
            if j == k:
                continue
            d = np.hypot(nx - x[j], ny - y[j])
            if d < nr + r[j] - 1e-9:
                ok = False
                break
        if not ok:
            continue
        new_val = cur_val + (nr - r[k])
        if new_val >= cur_val or rng.random() < np.exp((new_val - cur_val) / max(T, 1e-9)):
            x[k] = nx
            y[k] = ny
            r[k] = nr
            cur_val = new_val
            if cur_val > best_val:
                best_val = cur_val
                best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    return best_x, best_y, best_r, best_val


def construct():
    rng = np.random.default_rng(12345)
    # Start from grid packing
    grid = _initial_packing()
    x, y, r = grid[:, 0].copy(), grid[:, 1].copy(), grid[:, 2].copy()
    x, y, r = _make_valid(x, y, r)
    best_val = _sum_r(x, y, r)
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()

    # A few anneal restarts with decreasing temperature
    t0_list = [0.05, 0.04, 0.03, 0.02]
    t1 = 1e-4
    for t0 in t0_list:
        if time.time() > DEADLINE:
            break
        # start each restart from current best
        cx, cy, cr = best_x.copy(), best_y.copy(), best_r.copy()
        steps = 40000
        bx, by, br, bv = _anneal(cx, cy, cr, rng, steps, t0, t1)
        if bv > best_val:
            best_val = bv
            best_x, best_y, best_r = bx, by, br

    # Final validity safeguard
    if not _validity(best_x, best_y, best_r):
        bx, by, br = _make_valid(best_x.copy(), best_y.copy(), best_r.copy())
        if _validity(bx, by, br):
            best_x, best_y, best_r = bx, by, br

    return np.column_stack([best_x, best_y, best_r])
