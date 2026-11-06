"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 150.0 - 8.0  # safety margin before the 150s kill

def _radius(centers):
    """Vectorized max valid radius. Returns 0.0 if invalid."""
    if centers.shape != (N, 3) or not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    # boundary distance
    b = np.min(0.5 - np.abs(centers), axis=1)
    min_b = float(np.min(b))
    if min_b <= 0.0:
        return 0.0
    # pairwise distances via broadcasting
    diff = centers[:, None, :] - centers[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    # exclude self
    np.fill_diagonal(d, np.inf)
    min_p = float(np.min(d))
    if min_p <= 0.0:
        return 0.0
    return min(min_p / 2.0, min_b)

def _neg_radius(params):
    """Objective for minimize: negative radius (maximize)."""
    return -_radius(params.reshape(N, 3))

def _cubic_grid():
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _random_init(rng):
    margin = 0.1
    return rng.uniform(-0.5 + margin, 0.5 - margin, size=(N, 3))

def _local_optimize(centers, budget):
    """Run L-BFGS-B from a start; return best centers and radius."""
    t0 = time.time()
    params = centers.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)
    best_params = params.copy()
    best_r = _radius(centers)
    try:
        result = minimize(
            _neg_radius,
            params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-9, 'maxls': 50,
                     'maxfun': 5000},
        )
        if result.fun < -best_r:
            new_c = result.x.reshape(N, 3)
            new_r = _radius(new_c)
            if new_r > best_r:
                best_r = new_r
                best_params = result.x.copy()
    except Exception:
        pass
    return best_params.reshape(N, 3), best_r, time.time() - t0

def _anneal(centers, budget, rng):
    """Simulated annealing to escape local minima."""
    t0 = time.time()
    centers = centers.copy()
    cur_r = _radius(centers)
    best = centers.copy()
    best_r = cur_r
    T = 0.02
    cool = 0.99
    while time.time() - t0 < budget:
        if T < 1e-7:
            break
        idx = rng.randint(N)
        delta = rng.normal(0, T * 0.15, size=3)
        new_c = centers.copy()
        new_c[idx] += delta
        new_c[idx] = np.clip(new_c[idx], -0.5, 0.5)
        new_r = _radius(new_c)
        dr = new_r - cur_r
        if dr > 0 or rng.random() < np.exp(dr / T):
            centers = new_c
            cur_r = new_r
            if cur_r > best_r:
                best_r = cur_r
                best = new_c.copy()
        T *= cool
    return best, best_r, time.time() - t0

def construct():
    global_start = time.time()
    rng = np.random.default_rng(12345)

    best_centers = _cubic_grid()
    best_r = _radius(best_centers)

    def remaining():
        return DEADLINE - (time.time() - global_start)

    # 1) optimize from cubic grid
    if remaining() > 20.0:
        c, r, _ = _local_optimize(best_centers, min(40.0, remaining() - 10.0))
        if r > best_r:
            best_r, best_centers = r, c

    # 2) several random restarts with optimization
    for _ in range(6):
        if remaining() < 25.0:
            break
        c0 = _random_init(rng)
        c, r, _ = _local_optimize(c0, min(20.0, remaining() - 15.0))
        if r > best_r:
            best_r, best_centers = r, c

    # 3) anneal the best, then re-optimize
    if remaining() > 30.0:
        c, r, _ = _anneal(best_centers, min(25.0, remaining() - 20.0), rng)
        if r > best_r:
            best_r, best_centers = r, c
    if remaining() > 15.0:
        c, r, _ = _local_optimize(best_centers, min(12.0, remaining() - 8.0))
        if r > best_r:
            best_r, best_centers = r, c

    # final validity guard
    final = np.clip(best_centers, -0.5, 0.5)
    fr = _radius(final)
    if fr <= 0.0:
        final = _cubic_grid()
    return final
# EVOLVE-BLOCK-END
