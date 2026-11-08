"""Circle packing in unit square: pack 26 disjoint circles, maximize sum of radii."""
import numpy as np
import time

N = 26
_START = time.time()
DEADLINE = _START + 118.0          # safety margin before 150 s wall-clock kill
RNG = np.random.default_rng(20240517)

# --- force parameters (tuned for stability / speed) ---
K_WALL = 500.0
K_SEP = 40.0
K_R = 10.0
EPS = 1e-9
STEP = 0.03
MAXIT = 4000
R_MIN = 1e-6
R_MAX = 0.5


def _objective(c):
    """Sum of radii if the packing is valid, else -inf."""
    if c is None:
        return -np.inf
    x = c[:, 0]; y = c[:, 1]; r = c[:, 2]
    if (x.min() < -EPS) or (x.max() > 1 + EPS) or (y.min() < -EPS) or (y.max() > 1 + EPS):
        return -np.inf
    if r.min() < R_MIN:
        return -np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            if d < r[i] + r[j] - EPS:
                return -np.inf
    return float(np.sum(r))


def _forces(c):
    x = c[:, 0].copy(); y = c[:, 1].copy(); r = c[:, 2].copy()
    fx = np.zeros(N); fy = np.zeros(N); fr = np.zeros(N)

    # wall forces
    fx += K_WALL * np.where(x < r, 1.0, 0.0) - K_WALL * np.where(x > 1 - r, 1.0, 0.0)
    fy += K_WALL * np.where(y < r, 1.0, 0.0) - K_WALL * np.where(y > 1 - r, 1.0, 0.0)
    fr += K_WALL * (np.where(x < r, 1.0, 0.0) + np.where(x > 1 - r, 1.0, 0.0)
                    + np.where(y < r, 1.0, 0.0) + np.where(y > 1 - r, 1.0, 0.0))

    # separation forces
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]; dy = y[i] - y[j]
            d = np.hypot(dx, dy)
            s = r[i] + r[j]
            if d < s:
                if d < 1e-9:
                    dx = RNG.standard_normal(); dy = RNG.standard_normal()
                    d = np.hypot(dx, dy)
                ux = dx / d; uy = dy / d
                f = K_SEP * (s - d)
                fx[i] += f * ux; fy[i] += f * uy
                fx[j] -= f * ux; fy[j] -= f * uy
                fr[i] += f; fr[j] += f

    # maximize radii
    fr += K_R

    # clamp radii
    r = np.clip(r, R_MIN, R_MAX)
    return fx, fy, fr, r


def _optimize(c0):
    c = c0.copy()
    best = _objective(c)
    step = STEP
    for it in range(MAXIT):
        fx, fy, fr, r = _forces(c)
        c[:, 0] += step * fx
        c[:, 1] += step * fy
        c[:, 2] += step * fr
        c[:, 0] = np.clip(c[:, 0], 1e-6, 1 - 1e-6)
        c[:, 1] = np.clip(c[:, 1], 1e-6, 1 - 1e-6)
        c[:, 2] = np.clip(c[:, 2], R_MIN, R_MAX)
        val = _objective(c)
        if val > best:
            best = val
            step = min(step * 1.12, 0.12)
        else:
            step *= 0.82
            if it > 5:
                c = c0.copy()
            if step < 1e-4:
                break
        if time.time() > DEADLINE - 1.0:
            break
    return c, best


def _seed_grid():
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
             for i in range(side) for j in range(side)]
    return np.array(cells[:N])


def construct():
    t0 = time.time()
    c_grid = _seed_grid()
    best_c, best_val = _optimize(c_grid)

    # diverse random starts
    for _ in range(60):
        if time.time() > DEADLINE - 2.0:
            break
        x = RNG.uniform(0.08, 0.92, N)
        y = RNG.uniform(0.08, 0.92, N)
        r = RNG.uniform(0.02, 0.08, N)
        c0 = np.column_stack([x, y, r])
        cc, val = _optimize(c0)
        if val > best_val:
            best_val, best_c = val, cc

    # final polish on the best packing
    if time.time() < DEADLINE - 1.0:
        cc, val = _optimize(best_c)
        if val > best_val:
            best_val, best_c = val, cc

    # fallback: seed grid is always valid
    if best_val <= 0:
        return c_grid
    return best_c
