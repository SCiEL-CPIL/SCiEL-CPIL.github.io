"""Simulated annealing + local refinement for 50 equal circles in a unit square."""
import numpy as np
import time

N = 50
DEADLINE_MARGIN = 10.0  # seconds before the 150s kill


def _compute_radius(centers):
    """Compute the largest common radius for given centers."""
    # Wall distances
    wall = np.minimum(
        np.minimum(0.5 - centers[:, 0], 0.5 + centers[:, 0]),
        np.minimum(0.5 - centers[:, 1], 0.5 + centers[:, 1])
    )
    min_wall = np.min(wall)

    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    # Zero out diagonal
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0

    return min(min_wall, min_pair)


def _compute_radius_fast(centers):
    """Vectorized radius computation - avoids full NxN matrix when possible."""
    # Wall distances
    wall = np.minimum(
        np.minimum(0.5 - centers[:, 0], 0.5 + centers[:, 0]),
        np.minimum(0.5 - centers[:, 1], 0.5 + centers[:, 1])
    )
    min_wall = np.min(wall)

    # Pairwise: use scipy cdist if available, else manual
    try:
        from scipy.spatial.distance import cdist
        dist = cdist(centers, centers)
        np.fill_diagonal(dist, np.inf)
        min_pair = np.min(dist) / 2.0
    except ImportError:
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_pair = np.min(dist) / 2.0

    return min(min_wall, min_pair)


def _init_hexagonal(n, seed=42):
    """Initialize with a hexagonal-like pattern."""
    rng = np.random.default_rng(seed)
    # Start with a grid and add jitter
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])[:n]
    # Add small random jitter
    grid = grid + rng.uniform(-0.01, 0.01, size=grid.shape)
    # Clip to valid range
    grid = np.clip(grid, -0.45, 0.45)
    return grid


def _init_random(n, seed=42):
    """Random initialization inside the square."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-0.45, 0.45, size=(n, 2))
    return centers


def _simulated_annealing(centers, start_time, max_time, seed=42):
    """Run simulated annealing to maximize the minimum radius."""
    rng = np.random.default_rng(seed)
    best_centers = centers.copy()
    best_radius = _compute_radius_fast(centers)
    current_radius = best_radius

    n = len(centers)
    # Temperature schedule
    T_start = 0.05
    T_end = 0.0001
    # We'll iterate until time runs out
    while time.time() - start_time < max_time:
        # Number of iterations per batch
        batch = 200
        for _ in range(batch):
            if time.time() - start_time > max_time:
                break

            # Random perturbation: move one or a few circles
            n_move = rng.integers(1, 4)
            idx = rng.choice(n, size=n_move, replace=False)
            step_size = T_start * 0.5 * (current_radius / max(best_radius, 1e-10))
            step_size = max(step_size, 0.001)
            step_size = min(step_size, 0.1)

            old = centers[idx].copy()
            centers[idx] += rng.normal(0, step_size, size=(n_move, 2))
            # Clip to stay in bounds
            centers[idx] = np.clip(centers[idx], -0.49, 0.49)

            new_radius = _compute_radius_fast(centers)

            if new_radius > current_radius:
                current_radius = new_radius
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_centers = centers.copy()
            else:
                # Accept with probability exp((new - old) / T)
                delta = new_radius - current_radius
                # Annealing temperature decreases over time
                T = T_start * (T_end / T_start) ** ((time.time() - start_time) / max_time)
                T = max(T, T_end)
                if rng.random() < np.exp(delta / (T + 1e-12)):
                    current_radius = new_radius
                else:
                    centers[idx] = old

    return best_centers, best_radius


def _local_refinement(centers, start_time, max_time, seed=42):
    """Local refinement: iteratively push overlapping circles apart."""
    rng = np.random.default_rng(seed)
    best_centers = centers.copy()
    best_radius = _compute_radius_fast(centers)
    current = centers.copy()
    current_radius = best_radius

    n = len(current)

    # Iterative local optimization
    while time.time() - start_time < max_time:
        improved = False
        for i in range(n):
            if time.time() - start_time > max_time:
                break
            # Compute all distances from circle i to others and walls
            diff_others = current[i] - current
            dists = np.sqrt(np.sum(diff_others ** 2, axis=1))
            np.fill_diagonal(dists, np.inf)

            # Wall constraints
            wall_dists = np.array([
                0.5 - current[i, 0], 0.5 + current[i, 0],
                0.5 - current[i, 1], 0.5 + current[i, 1]
            ])

            # Find the most constraining neighbor/wall
            min_dist = np.min(dists)
            min_wall = np.min(wall_dists)

            # Determine direction to move circle i
            # Move away from the closest constraint
            if min_dist < min_wall + 1e-15:
                # Move away from closest circle
                j = np.argmin(dists)
                direction = current[i] - current[j]
                norm = np.linalg.norm(direction)
                if norm > 1e-15:
                    direction /= norm
            else:
                # Move away from closest wall
                if np.argmin(wall_dists) == 0:
                    direction = np.array([1.0, 0.0])
                elif np.argmin(wall_dists) == 1:
                    direction = np.array([-1.0, 0.0])
                elif np.argmin(wall_dists) == 2:
                    direction = np.array([0.0, 1.0])
                else:
                    direction = np.array([0.0, -1.0])

            # Try several step sizes
            best_step = 0
            best_local_r = current_radius
            for step in [0.01, 0.005, 0.002, 0.001, 0.0005]:
                trial = current.copy()
                trial[i] = current[i] + direction * step
                trial[i] = np.clip(trial[i], -0.49, 0.49)
                r = _compute_radius_fast(trial)
                if r > best_local_r:
                    best_local_r = r
                    best_step = step

            if best_step > 0:
                current[i] += direction * best_step
                current[i] = np.clip(current[i], -0.49, 0.49)
                current_radius = best_local_r
                improved = True

                if current_radius > best_radius:
                    best_radius = current_radius
                    best_centers = current.copy()

        if not improved:
            # Try random small kicks to escape local optima
            kick_idx = rng.integers(0, n)
            kick_dir = rng.normal(0, 1, size=2)
            kick_dir /= np.linalg.norm(kick_dir)
            current[kick_idx] += kick_dir * 0.02
            current[kick_idx] = np.clip(current[kick_idx], -0.49, 0.49)
            current_radius = _compute_radius_fast(current)

    return best_centers, best_radius


def construct():
    start_time = time.time()
    max_time = 150.0 - DEADLINE_MARGIN  # 140s budget

    best_centers = None
    best_radius = -np.inf

    # Multiple restarts with different initializations
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]

    for seed in seeds:
        elapsed = time.time() - start_time
        remaining = max_time - elapsed
        if remaining < 5.0:
            break

        # Choose initialization
        if seed % 2 == 0:
            centers = _init_hexagonal(N, seed=seed)
        else:
            centers = _init_random(N, seed=seed)

        # Phase 1: Simulated annealing
        sa_time = min(remaining * 0.6, 30.0)
        centers, r = _simulated_annealing(centers, time.time(), sa_time, seed=seed)

        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()

        # Phase 2: Local refinement
        elapsed = time.time() - start_time
        remaining = max_time - elapsed
        if remaining < 3.0:
            break

        lr_time = remaining - 2.0
        centers2, r2 = _local_refinement(centers, time.time(), lr_time, seed=seed)

        if r2 > best_radius:
            best_radius = r2
            best_centers = centers2.copy()

    # Final safety check
    if best_centers is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_centers = np.array([(x, y) for y in ticks for x in ticks])[:N]

    # Ensure validity: clip to square
    best_centers = np.clip(best_centers, -0.5, 0.5)

    return best_centers
