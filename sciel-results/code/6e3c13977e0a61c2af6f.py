"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START

# Hard wall-clock budget (evaluator kills at 150s); keep a safety margin.
DEADLINE = time.time() + 142.0


def _compute_radius(pts):
    """Exact maximum common radius for given centers (double precision)."""
    if len(pts) < 2:
        return 0.0
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    dx = 0.5 - np.abs(pts[:, 0])
    dy = 0.5 - np.abs(pts[:, 1])
    min_boundary = np.min(np.minimum(dx, dy))
    return min(min_dist / 2.0, min_boundary)


def _softmin(vals, beta):
    """Smooth approximation of min(vals) via log-sum-exp."""
    m = np.min(vals)
    return m - np.log(np.sum(np.exp(-beta * (vals - m)))) / beta


def _objective(x, beta):
    """Smooth surrogate to MINIMIZE: -softmin(min_pair_dist/2, boundary)."""
    pts = x.reshape(-1, 2)

    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, np.inf)
    pair = 0.5 * np.sqrt(d2)  # half pairwise distances (candidate radii)

    dx = 0.5 - np.abs(pts[:, 0])
    dy = 0.5 - np.abs(pts[:, 1])
    wall = np.minimum(dx, dy)  # boundary distances (candidate radii)

    allv = np.concatenate([pair.ravel(), wall.ravel()])
    return -_softmin(allv, beta)


def _initial_hex():
    """Hexagonal (staggered) packing of N points inside [-0.5, 0.5]^2."""
    s = 0.11
    h = s * np.sqrt(3.0) / 2.0
    rows = []
    y = -0.5 + 0.55 * s
    i = 0
    while y <= 0.5 - 0.55 * s + 1e-9:
        offset = 0.0 if (i % 2 == 0) else s / 2.0
        x = -0.5 + 0.55 * s + offset
        j = 0
        while x <= 0.5 - 0.55 * s + 1e-9:
            rows.append([x, y])
            x += s
            j += 1
        y += h
        i += 1
    arr = np.array(rows, dtype=np.float64)
    if len(arr) < N:
        # pad with a fine grid if the hex rows are too short
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks], dtype=np.float64)
        arr = np.vstack([arr, grid])
    arr = arr[:N]
    arr = np.clip(arr, -0.5 + 1e-9, 0.5 - 1e-9)
    return arr


def _simulated_annealing(pts, time_budget):
    """Global search via simulated annealing on the smooth surrogate."""
    rng = np.random.default_rng(12345)
    x = pts.copy()
    best_x = x.copy()
    best_val = _objective(x, 400.0)
    start = time.time()
    T = 0.03
    T_min = 1e-5
    n = len(x)
    while T > T_min and time.time() < start + time_budget:
        step = rng.normal(0, T, size=(1, 2))
        new_x = x + step
        new_x = np.clip(new_x, -0.5, 0.5)
        f_old = _objective(x, 400.0)
        f_new = _objective(new_x, 400.0)
        d = f_new - f_old
        if d < 0 or rng.random() < np.exp(-d / T):
            x = new_x
            if f_new < best_val:
                best_val = f_new
                best_x = x.copy()
        T *= 0.995
        if n % 400 == 0:
            pass
    return best_x


def _polish(pts, time_budget):
    """High-precision gradient polish on the smooth surrogate (L-BFGS-B)."""
    start = time.time()
    bounds = [(-0.5 + 1e-9, 0.5 - 1e-9)] * (2 * N)
    best_x = pts.copy()
    best_val = _objective(pts, 800.0)
    for beta in (400.0, 800.0, 1500.0):
        if time.time() > start + time_budget:
            break
        try:
            res = minimize(_objective, best_x, args=(beta,), method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-9})
            if _objective(res.x, beta) < best_val:
                best_val = _objective(res.x, beta)
                best_x = res.x.copy()
        except Exception:
            pass
    return np.clip(best_x, -0.5 + 1e-9, 0.5 - 1e-9)


def construct():
    """Construct centers of N equal circles inside the square [-1/2, 1/2]^2."""
    start_time = time.time()
    limit = 140.0  # safety margin before the 150s kill

    pts = _initial_hex()
    best_pts = pts.copy()
    best_radius = _compute_radius(pts)

    # Phase 1: global search (simulated annealing)
    if time.time() - start_time < limit - 40:
        try:
            sa = _simulated_annealing(pts, time_budget=min(50.0, limit - (time.time() - start_time) - 30))
            r = _compute_radius(sa)
            if r > best_radius:
                best_radius = r
                best_pts = sa.copy()
        except Exception:
            pass

    # Phase 2: gradient polish on the best configuration found
    if time.time() - start_time < limit - 5:
        try:
            pol = _polish(best_pts, time_budget=max(1.0, limit - (time.time() - start_time) - 3))
            r = _compute_radius(pol)
            if r > best_radius:
                best_radius = r
                best_pts = pol.copy()
        except Exception:
            pass

    # Final validity safety: clip and re-verify
    best_pts = np.clip(best_pts, -0.5 + 1e-9, 0.5 - 1e-9)
    return best_pts
# EVOLVE-BLOCK-END
