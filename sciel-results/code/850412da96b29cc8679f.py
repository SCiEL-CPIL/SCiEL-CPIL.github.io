"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Grid initialization
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    grid = grid[:N]

    # Random initialization
    rng = np.random.default_rng(42)
    rand = rng.uniform(-0.5, 0.5, size=(N, 2))

    # Hybrid: use grid as primary, but also try random
    candidates = [grid.copy(), rand.copy()]

    best_r = -1
    best_centers = None

    for centers0 in candidates:
        # Local refinement using scipy minimize on negative radius objective
        # We want to maximize min distance / 2 subject to staying in square
        # Use a smooth approximation: minimize sum of exp(-lambda * dist) or similar
        # Actually, let's use a direct approach: maximize the minimum pair distance
        # via coordinate descent or scipy's SLSQP

        # First, do a simple coordinate descent to improve
        centers = _coordinate_descent(centers0, max_iters=2000, deadline=deadline)
        if centers is None:
            continue

        # Then polish with scipy
        centers_polished = _scipy_polish(centers, deadline=deadline)
        if centers_polished is not None:
            r = _compute_radius(centers_polished)
            if r > best_r:
                best_r = r
                best_centers = centers_polished.copy()

        if time.time() > deadline - 5.0:
            break

    if best_centers is None:
        return grid

    return best_centers


def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    n = len(centers)
    # Wall distances
    wall_dists = np.min(np.array([
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    ]), axis=0)
    min_wall = np.min(wall_dists)

    # Pair distances
    if n > 1:
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
    else:
        min_pair = 0.5

    return min(min_wall, min_pair)


def _coordinate_descent(centers, max_iters=2000, deadline=None):
    """Simple coordinate descent to improve the minimum radius."""
    centers = centers.copy()
    n = len(centers)
    current_r = _compute_radius(centers)

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 2.0:
            break

        improved = False
        for i in range(n):
            for dim in range(2):
                # Try moving point i in dimension dim
                best_x = centers[i, dim]
                best_local_r = current_r

                # Try several step sizes
                for step in [0.01, 0.005, 0.002, 0.001]:
                    for direction in [-1, 1]:
                        new_x = centers[i, dim] + direction * step
                        # Check bounds
                        if new_x < -0.5 or new_x > 0.5:
                            continue
                        old_val = centers[i, dim]
                        centers[i, dim] = new_x
                        r = _compute_radius(centers)
                        centers[i, dim] = old_val

                        if r > best_local_r + 1e-12:
                            best_local_r = r
                            best_x = new_x

                if best_x != centers[i, dim]:
                    centers[i, dim] = best_x
                    current_r = best_local_r
                    improved = True

        if not improved:
            break

    return centers


def _scipy_polish(centers, deadline=None):
    """Polish using scipy minimize with a smooth objective."""
    if deadline is not None and time.time() > deadline - 5.0:
        return centers

    n = len(centers)
    x0 = centers.flatten()

    def objective(x):
        if deadline is not None and time.time() > deadline - 3.0:
            return 0.0
        c = x.reshape(n, 2)
        # Soft minimum of distances
        # Use log-sum-exp for smooth min
        # Pair distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        # Wall distances
        wall_dists = np.min(np.array([
            0.5 - np.abs(c[:, 0]),
            0.5 - np.abs(c[:, 1])
        ]), axis=0)

        # Smooth min using log-sum-exp
        beta = 100.0
        # For pairs
        flat_dists = dists[~np.eye(n, dtype=bool)]
        # Soft min
        min_pair_soft = -np.log(np.sum(np.exp(-beta * flat_dists))) / beta
        # Walls
        min_wall_soft = -np.log(np.sum(np.exp(-beta * wall_dists))) / beta

        # We want to maximize the min, so minimize the negative
        return -min(min_pair_soft, min_wall_soft)

    # Bounds
    bounds = [(-0.5, 0.5)] * (2 * n)

    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 500, 'ftol': 1e-12})
        if result.success or result.fun < 0:
            polished = result.x.reshape(n, 2)
            r_polished = _compute_radius(polished)
            r_orig = _compute_radius(centers)
            if r_polished >= r_orig - 1e-10:
                return polished
    except Exception:
        pass

    return centers

# EVOLVE-BLOCK-END
