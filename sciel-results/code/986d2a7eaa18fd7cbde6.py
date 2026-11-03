"""Structural change: Nelder-Mead on exact min-max radius from hex lattice + local polish."""
N = 200
import numpy as np
import time

_DEADLINE = 145.0  # Safety margin below 150s kill
_START_TIME = time.time()


def _check_time():
    return time.time() - _START_TIME < _DEADLINE


def _compute_radius(centers):
    """Return the largest common radius for given centers in [-0.5,0.5]^2."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Boundary constraint
    boundary = np.min(np.minimum(
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    ))
    if boundary < 0:
        return 0.0
    # Pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    return min(boundary, min_pair / 2.0)


def _negative_radius(centers_flat):
    """Objective to minimize: negative of the radius."""
    centers = centers_flat.reshape(N, 2)
    return -_compute_radius(centers)


def _generate_hex_initial(n_points, spacing_factor=1.0):
    """Generate a hexagonal grid of points with given spacing factor."""
    # Base spacing for N points in unit square
    s_base = np.sqrt(2.0 / (N * np.sqrt(3)))
    s = s_base * spacing_factor

    # Generate points
    points = []
    row_height = s * np.sqrt(3) / 2
    n_rows = int(1.0 / row_height) + 4
    n_cols = int(1.0 / s) + 4

    # Center the grid
    total_width = (n_cols - 1) * s
    total_height = (n_rows - 1) * row_height

    start_x = -0.5 + (1.0 - total_width) / 2
    start_y = -0.5 + (1.0 - total_height) / 2

    for i in range(n_rows):
        for j in range(n_cols):
            x = start_x + j * s
            if i % 2 == 1:
                x += s / 2
            y = start_y + i * row_height
            if -0.495 <= x <= 0.495 and -0.495 <= y <= 0.495:
                points.append([x, y])

    points = np.array(points)

    if len(points) > n_points:
        center_dist = np.sum(points ** 2, axis=1)
        idx = np.argsort(center_dist)[:n_points]
        points = points[idx]
    elif len(points) < n_points:
        extra = np.random.uniform(-0.45, 0.45, size=(n_points - len(points), 2))
        points = np.vstack([points, extra])

    return points[:n_points]


def _local_polish(centers, max_iters=200):
    """Iterative local search: move the most constrained point away from its constraint."""
    r_best = _compute_radius(centers)
    for _ in range(max_iters):
        if time.time() - _START_TIME > _DEADLINE - 3:
            break

        diff = centers[:, None, :] - centers[None, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_margins = dists / 2.0

        bound_x = 0.5 - np.abs(centers[:, 0])
        bound_y = 0.5 - np.abs(centers[:, 1])
        bound_margins = np.minimum(bound_x, bound_y)

        min_pair_per_point = np.min(pair_margins, axis=1)
        min_margin_per_point = np.minimum(min_pair_per_point, bound_margins)

        idx_min = np.argmin(min_margin_per_point)

        improved = False
        if min_pair_per_point[idx_min] < bound_margins[idx_min]:
            j = np.argmin(pair_margins[idx_min])
            vec = centers[idx_min] - centers[j]
            norm = np.linalg.norm(vec)
            if norm > 1e-10:
                vec /= norm
                for step in [0.002, 0.001, 0.0005]:
                    new_c = centers.copy()
                    new_c[idx_min] += vec * step
                    new_c[idx_min] = np.clip(new_c[idx_min], -0.495, 0.495)
                    r_new = _compute_radius(new_c)
                    if r_new > r_best:
                        centers = new_c
                        r_best = r_new
                        improved = True
                        break
        else:
            vec = -centers[idx_min]
            norm = np.linalg.norm(vec)
            if norm > 1e-10:
                vec /= norm
                for step in [0.002, 0.001, 0.0005]:
                    new_c = centers.copy()
                    new_c[idx_min] += vec * step
                    new_c[idx_min] = np.clip(new_c[idx_min], -0.495, 0.495)
                    r_new = _compute_radius(new_c)
                    if r_new > r_best:
                        centers = new_c
                        r_best = r_new
                        improved = True
                        break

        if not improved:
            break

    return centers, r_best


def construct():
    from scipy.optimize import minimize

    start_time = time.time()
    best_centers = None
    best_r = 0.0

    # Try multiple diverse initializations
    init_configs = [
        (1.0, 0.001),   # standard hex with small noise
        (1.1, 0.002),   # slightly spaced hex
        (0.95, 0.001),  # denser hex
        (1.05, 0.003),  # sparser hex with more noise
    ]

    for idx, (spac, noise) in enumerate(init_configs):
        if time.time() - start_time > _DEADLINE - 10:
            break

        centers = _generate_hex_initial(N, spac)
        centers += np.random.normal(0, noise, size=centers.shape)
        centers = np.clip(centers, -0.495, 0.495)

        r_init = _compute_radius(centers)
        if r_init <= 0:
            continue

        # Run Nelder-Mead on the exact radius objective
        # We optimize in the 400-dimensional space of centers
        # Use a moderate initial step size
        try:
            result = minimize(
                _negative_radius,
                centers.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 300,
                    'xatol': 1e-6,
                    'fatol': 1e-7,
                    'adaptive': True,
                    'initial_simplex': None  # let scipy choose
                }
            )
        except Exception:
            result = None

        if result is not None and result.fun < -r_init:
            new_centers = result.x.reshape(N, 2)
            new_centers = np.clip(new_centers, -0.495, 0.495)
            r_new = _compute_radius(new_centers)
            if r_new > best_r:
                best_r = r_new
                best_centers = new_centers.copy()
        else:
            if r_init > best_r:
                best_r = r_init
                best_centers = centers.copy()

    if best_centers is None:
        # Fallback grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
        best_r = _compute_radius(best_centers)

    # Final local polish
    if time.time() - start_time < _DEADLINE - 2:
        best_centers, best_r = _local_polish(best_centers, max_iters=300)

    # Final validation
    r_final = _compute_radius(best_centers)
    if r_final <= 0:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]

    return best_centers
