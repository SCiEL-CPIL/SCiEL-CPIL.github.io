"""Stochastic local search with simulated annealing for packing 100 equal spheres in a unit cube."""
import numpy as np

N = 100
HALF = 0.5
DEADLINE_MARGIN = 5.0  # seconds of safety margin before the 150s kill


def _radius(pts):
    """Compute the feasible common radius for a set of centres (vectorized)."""
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    # pairwise differences
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, np.inf)
    min_pair = np.sqrt(np.min(d2))
    # wall distances
    wall = HALF - np.max(np.abs(pts), axis=1)
    min_wall = np.min(wall)
    r = 0.5 * min(min_pair, 2.0 * min_wall)
    if not np.isfinite(r) or r < 0:
        return 0.0
    return float(r)


def _init_grid():
    """Simple cubic grid initialization (valid, gives a decent starting radius)."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].astype(np.float64)


def _init_random(rng):
    """Random initialization well inside the cube."""
    return (rng.random((N, 3)) - 0.5) * 0.9


def construct():
    import time
    t0 = time.time()
    deadline = 150.0 - DEADLINE_MARGIN

    rng = np.random.default_rng(12345)

    # Best-so-far state
    best_pts = _init_grid()
    best_r = _radius(best_pts)

    # Try a few initializations to get a good starting point for SA
    for seed in range(6):
        if time.time() > deadline:
            break
        pts = _init_random(rng)
        r = _radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    current = best_pts.copy()
    current_r = best_r

    # Simulated annealing schedule
    T0 = 0.05
    Tend = 1e-6
    n_iter_max = 200000

    for it in range(n_iter_max):
        if time.time() > deadline:
            break

        # Temperature (geometric cooling)
        frac = it / max(1, n_iter_max)
        T = T0 * (Tend / T0) ** frac

        # Choose a random sphere to move
        idx = rng.integers(N)
        old = current[idx]

        # Random perturbation: move by a small random step
        step_scale = 0.05 * (0.3 + 0.7 * T / T0)
        new = old + rng.standard_normal(3) * step_scale

        # Clamp to keep inside the cube (with a small margin so wall term stays valid)
        new = np.clip(new, -HALF, HALF)

        current[idx] = new
        new_r = _radius(current)

        delta = new_r - current_r
        if delta >= 0 or rng.random() < np.exp(delta / max(T, 1e-12)):
            # accept
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_pts = current.copy()
        else:
            # reject, revert
            current[idx] = old

        # Occasionally do a larger random re-seed of one sphere to escape deep minima
        if rng.random() < 0.005:
            current[idx] = (rng.random(3) - 0.5) * 0.9
            current_r = _radius(current)
            if current_r > best_r:
                best_r = current_r
                best_pts = current.copy()

    return best_pts
