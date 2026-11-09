"""Packing 50 circles in unit square via force-directed relaxation with annealing."""
import numpy as np
import time

N = 50
_DEADLINE = 145.0  # safety margin before 150s kill


def _radius_score(centers):
    """Compute the largest common radius for given centers."""
    n = len(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0

    dist_to_boundary = np.min(np.stack([
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    ], axis=1), axis=1)
    min_boundary = np.min(dist_to_boundary)

    return min(min_pair, min_boundary)


def _hex_init(n_circles):
    """Hexagonal packing initialization."""
    for n_rows in range(4, 12):
        n_cols = int(np.ceil(n_circles / n_rows))
        x_spacing = 1.0 / (n_cols + 0.5)
        y_spacing = x_spacing * np.sqrt(3) / 2
        total_height = (n_rows - 1) * y_spacing
        if total_height > 0.9:
            continue
        centers = []
        for row in range(n_rows):
            if row % 2 == 0:
                n_in_row = min(n_cols, n_circles - len(centers))
                offset = 0
            else:
                n_in_row = min(n_cols, n_circles - len(centers))
                offset = x_spacing / 2
            for col in range(n_in_row):
                x = -0.5 + x_spacing / 2 + col * x_spacing + offset
                y = -0.5 + y_spacing / 2 + row * y_spacing
                if x < 0.49 and y < 0.49:
                    centers.append([x, y])
        if len(centers) >= n_circles:
            return np.array(centers[:n_circles])
    m = int(np.ceil(np.sqrt(n_circles)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:n_circles]


def _grid_init(n_circles, jitter=0.0):
    m = int(np.ceil(np.sqrt(n_circles)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    result = grid[:n_circles]
    if jitter > 0:
        result = result + np.random.randn(*result.shape) * jitter
        result = np.clip(result, -0.49, 0.49)
    return result


def _random_init(n_circles):
    return np.random.uniform(-0.45, 0.45, size=(n_circles, 2))


def _force_directed(centers, start_time, deadline, max_steps=50000):
    """
    Force-directed relaxation:
    - Each circle repels others with force ~ 1/d^2 (strong at close range)
    - Walls exert restoring force when circle approaches boundary
    - Step size anneals geometrically
    - Periodic "kick" to escape local minima
    """
    centers = centers.copy().astype(np.float64)
    n = len(centers)
    best_r = _radius_score(centers)
    best_centers = centers.copy()

    # Annealing schedule
    T0 = 0.01  # initial step size
    T_end = 1e-8
    cool_rate = (T_end / T0) ** (1.0 / max_steps)

    T = T0
    step = 0

    while step < max_steps:
        if time.time() > deadline:
            break

        # Compute forces
        forces = np.zeros_like(centers)

        # Pairwise repulsion: F_ij = (1/d^2) * (c_i - c_j)/d  for d < 2*r_target
        # We use a target radius to scale forces
        r_target = best_r * 1.01  # slightly above current best to push apart

        for i in range(n):
            for j in range(i + 1, n):
                d_vec = centers[i] - centers[j]
                d = np.sqrt(np.dot(d_vec, d_vec))
                if d < 1e-12:
                    # Random kick for coincident points
                    d_vec = np.random.randn(2) * 1e-6
                    d = np.sqrt(np.dot(d_vec, d_vec))
                # Force magnitude: strong repulsion when d < 2*r_target
                # Use inverse-square for close, linear for far
                if d < 2 * r_target:
                    # Overlap region: strong repulsion
                    f_mag = (2 * r_target - d) / (d * d + 1e-10)
                else:
                    # Gentle repulsion to maintain spacing
                    f_mag = 0.1 / (d * d + 1e-10)

                f_vec = f_mag * d_vec / (d + 1e-12)
                forces[i] += f_vec
                forces[j] -= f_vec

        # Wall forces: push away from boundary
        for i in range(n):
            # x-boundary
            if centers[i, 0] > 0.5 - r_target:
                forces[i, 0] -= (centers[i, 0] - (0.5 - r_target)) / (r_target + 1e-10)
            if centers[i, 0] < -0.5 + r_target:
                forces[i, 0] += ((-0.5 + r_target) - centers[i, 0]) / (r_target + 1e-10)
            # y-boundary
            if centers[i, 1] > 0.5 - r_target:
                forces[i, 1] -= (centers[i, 1] - (0.5 - r_target)) / (r_target + 1e-10)
            if centers[i, 1] < -0.5 + r_target:
                forces[i, 1] += ((-0.5 + r_target) - centers[i, 1]) / (r_target + 1e-10)

        # Apply forces with annealed step size
        centers += T * forces

        # Clip to valid range
        centers = np.clip(centers, -0.499, 0.499)

        # Cool down
        T *= cool_rate

        # Periodic kick to escape local minima
        if step % 500 == 499:
            kick = np.random.randn(*centers.shape) * T * 0.5
            centers += kick
            centers = np.clip(centers, -0.499, 0.499)

        # Update best
        r = _radius_score(centers)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()

        step += 1

    return best_centers, best_r


def _local_refinement(centers, start_time, deadline):
    """
    Final refinement: coordinate descent + small random perturbations
    on the constraining pairs.
    """
    centers = centers.copy().astype(np.float64)
    n = len(centers)
    best_r = _radius_score(centers)
    best_centers = centers.copy()

    # Find constraining pairs (those at minimum distance)
    def _get_constraining_pairs(c, r):
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(c[i] - c[j])
                if d < 2 * r * 1.001:
                    pairs.append((i, j, d))
        # Also check boundary constraints
        for i in range(n):
            for k in range(2):
                if k == 0:
                    d = min(0.5 - c[i, 0], c[i, 0] + 0.5)
                else:
                    d = min(0.5 - c[i, 1], c[i, 1] + 0.5)
                if d < r * 1.001:
                    pairs.append((i, -1, d))
        return pairs

    # Coordinate descent with small steps
    step_size = 0.001
    n_iter = 0
    max_iter = 200000

    while n_iter < max_iter:
        if time.time() > deadline:
            break

        improved = False

        # Try small random perturbations on random circle
        i = np.random.randint(n)
        orig = centers[i].copy()
        for _ in range(10):
            pert = np.random.randn(2) * step_size
            centers[i] = np.clip(orig + pert, -0.499, 0.499)
            r = _radius_score(centers)
            if r > best_r:
                best_r = r
                best_centers = centers.copy()
                improved = True
            else:
                centers[i] = orig
                break

        # Try moving along axis
        for axis in range(2):
            for direction in [1, -1]:
                orig = centers[i].copy()
                centers[i, axis] += direction * step_size
                centers[i] = np.clip(centers[i], -0.499, 0.499)
                r = _radius_score(centers)
                if r > best_r:
                    best_r = r
                    best_centers = centers.copy()
                    improved = True
                else:
                    centers[i] = orig

        if not improved and n_iter % 1000 == 0:
            # Reduce step size
            step_size *= 0.95
            if step_size < 1e-9:
                break

        n_iter += 1

    return best_centers, best_r


def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE

    best_centers = None
    best_val = -np.inf

    # Generate diverse initializations
    inits = []
    inits.append(_hex_init(N))
    inits.append(_grid_init(N))
    for _ in range(3):
        inits.append(_grid_init(N, jitter=0.03))
    for _ in range(4):
        inits.append(_random_init(N))

    # Force-directed relaxation from each initialization
    for idx, init_c in enumerate(inits):
        if time.time() > deadline - 30:
            break

        remaining = deadline - time.time()
        # Allocate time proportional to remaining budget
        time_per_init = min(remaining / (len(inits) - idx), 25.0)
        init_deadline = time.time() + time_per_init

        centers, val = _force_directed(init_c, start_time, init_deadline)
        if val > best_val:
            best_val = val
            best_centers = centers

    # Local refinement on best result
    if best_centers is not None and time.time() < deadline - 5:
        centers, val = _local_refinement(best_centers, start_time, deadline)
        if val > best_val:
            best_val = val
            best_centers = centers

    # If we have more time, do additional random restarts with force-directed
    while time.time() < deadline - 10:
        r_c = _random_init(N)
        init_deadline = min(time.time() + 15.0, deadline - 5)
        centers, val = _force_directed(r_c, start_time, init_deadline)
        if val > best_val:
            best_val = val
            best_centers = centers

        # Quick local refinement
        if time.time() < deadline - 5:
            centers2, val2 = _local_refinement(best_centers, start_time, deadline)
            if val2 > best_val:
                best_val = val2
                best_centers = centers2

    if best_centers is None:
        best_centers = _grid_init(N)

    # Ensure valid output
    best_centers = np.clip(best_centers, -0.5, 0.5)

    return best_centers
