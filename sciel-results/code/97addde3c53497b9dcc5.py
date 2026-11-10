import numpy as np
import time
import sys
from scipy.spatial.distance import cdist
from scipy.spatial import Delaunay
from scipy.optimize import minimize

N = 200
TIME_LIMIT = 140.0
START_TIME = time.time()


def _check_time():
    return time.time() - START_TIME


def _compute_radius(centres):
    """Compute the packing radius. Returns r if valid, 0.0 if invalid."""
    if len(centres) != N:
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    if np.any(centres < -0.5) or np.any(centres > 0.5):
        return 0.0

    dx = np.abs(centres[:, 0])
    dy = np.abs(centres[:, 1])
    wall_dists = np.min(np.stack([0.5 - dx, 0.5 - dy], axis=1), axis=1)
    min_wall = np.min(wall_dists)
    if min_wall <= 0:
        return 0.0

    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    return min(min_dist / 2.0, min_wall)


def _generate_hex_init():
    """
    Generate a hexagonal lattice initialization, selecting the best subset of N points.
    """
    best_centres = None
    best_r = -1

    # Try multiple hex spacings
    for s in np.linspace(0.036, 0.046, 50):
        h_hex = s * np.sqrt(3) / 2.0
        points = []
        num_rows = int(0.5 / h_hex) + 3
        for j in range(-num_rows, num_rows + 1):
            y = j * h_hex
            if abs(y) > 0.5:
                continue
            offset = (j % 2) * s / 2.0
            num_cols = int(0.5 / s) + 3
            for i in range(-num_cols, num_cols + 1):
                x = i * s + offset
                if -0.5 <= x <= 0.5:
                    points.append([x, y])

        points = np.array(points)
        if len(points) < N:
            continue

        # Select N points: prefer central points
        dist_to_center = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
        order = np.argsort(dist_to_center)
        selected = points[order[:N]]

        # Center the selection
        selected = selected - selected.mean(axis=0)

        # Check bounds
        if np.any(selected < -0.5) or np.any(selected > 0.5):
            continue

        r = _compute_radius(selected)
        if r > best_r:
            best_r = r
            best_centres = selected.copy()

    # Fallback: 20x10 grid with hex distortion
    if best_centres is None:
        w, h = 20, 10
        xs = (np.arange(w) - (w - 1) / 2.0)
        ys = (np.arange(h) - (h - 1) / 2.0)
        scale = 1.0 / (w - 1)
        grid = np.array([(x * scale, y * scale) for y in ys for x in xs])
        for i in range(h):
            if i % 2 != 0:
                grid[i * w:(i + 1) * w, 0] += scale / 2.0
        max_x = np.max(np.abs(grid[:, 0]))
        max_y = np.max(np.abs(grid[:, 1]))
        if max_x > 0.5 or max_y > 0.5:
            scale_down = 0.5 / max(max_x, max_y)
            grid *= scale_down
        best_centres = grid
        best_r = _compute_radius(grid)

    return best_centres, best_r


def _energy_and_grad(x, k=10.0):
    """
    Compute the energy and gradient for the soft repulsive potential.
    x is a flat array of length 2N.
    """
    centres = x.reshape(-1, 2)
    n = len(centres)

    # Pairwise distances
    diff = centres[:, None, :] - centres[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)

    # Wall distances
    wall = np.min(np.stack([
        0.5 - np.abs(centres[:, 0]),
        0.5 - np.abs(centres[:, 1])
    ], axis=1), axis=1)

    # Soft min of pair distances and wall distances
    a = dist / 2.0  # half of pair distance
    b = wall[None, :]  # wall distance for each point

    # We want to maximize the minimum clearance
    # Energy = -sum(softmin(clearances))
    # softmin(a, b) = -k * log(exp(-a/k) + exp(-b/k))

    # For each point i, compute softmin over all j != i and wall
    # This is expensive, so let's use a simpler formulation:
    # Energy = sum_i (-softmin_i) where softmin_i = softmin(min_pair_i, wall_i)

    min_pair = np.min(a, axis=1)  # min over j for each i
    # softmin of min_pair and wall
    Ea = np.exp(-min_pair / k)
    Eb = np.exp(-wall / k)
    E = Ea + Eb
    softmin_val = -k * np.log(E)

    energy = -np.sum(softmin_val)

    # Gradient
    # d(softmin_val)/d(min_pair) = Ea/E
    # d(softmin_val)/d(wall) = Eb/E
    wa = Ea / E
    wb = Eb / E

    grad = np.zeros_like(centres)

    for i in range(n):
        # Gradient from wall term
        if wb[i] > 1e-12:
            dx = centres[i, 0]
            dy = centres[i, 1]
            d_left = dx + 0.5
            d_right = 0.5 - dx
            d_down = dy + 0.5
            d_up = 0.5 - dy
            min_w = min(d_left, d_right, d_down, d_up)
            if min_w == d_left:
                grad[i, 0] += wb[i]
            elif min_w == d_right:
                grad[i, 0] -= wb[i]
            elif min_w == d_down:
                grad[i, 1] += wb[i]
            else:
                grad[i, 1] -= wb[i]

        # Gradient from pair term
        if wa[i] > 1e-12:
            j_star = np.argmin(dist[i])
            d_ij = dist[i, j_star]
            if d_ij < 1e-9:
                continue
            dir_vec = (centres[i] - centres[j_star]) / d_ij
            grad[i] += wa[i] * dir_vec
            grad[j_star] -= wa[i] * dir_vec

    return energy, grad


def _local_optimize(centres, time_limit):
    """
    Optimize using L-BFGS-B with a soft repulsive potential.
    """
    best_centres = centres.copy()
    best_r = _compute_radius(best_centres)

    deadline = time.time() + time_limit
    k = 10.0
    x0 = best_centres.ravel()

    # Use L-BFGS-B to minimize energy (which is negative of softmin)
    while time.time() < deadline:
        # Check time
        if time.time() > deadline - 5:
            break

        # Run L-BFGS-B
        result = minimize(
            _energy_and_grad,
            x0,
            args=(k,),
            jac=True,
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
        )

        new_centres = result.x.reshape(-1, 2)
        new_r = _compute_radius(new_centres)

        if new_r > best_r + 1e-12:
            best_r = new_r
            best_centres = new_centres
            x0 = new_centres.ravel()
            # Decrease k for finer optimization
            k *= 0.8
        else:
            # No improvement, try with smaller k
            k *= 0.7
            if k < 0.1:
                break

    return best_centres, best_r


def _delaunay_polish(centres, time_limit):
    """
    Polish using Delaunay triangulation and force balance.
    """
    best_centres = centres.copy()
    best_r = _compute_radius(best_centres)
    deadline = time.time() + time_limit

    step = 1e-4
    while time.time() < deadline and step > 1e-8:
        # Compute clearances
        dists = cdist(best_centres, best_centres)
        np.fill_diagonal(dists, np.inf)
        min_neighbor_dists = np.min(dists, axis=1)

        wall_clearances = np.min(np.stack([
            0.5 - np.abs(best_centres[:, 0]),
            0.5 - np.abs(best_centres[:, 1])
        ], axis=1), axis=1)

        clearances = np.minimum(min_neighbor_dists / 2.0, wall_clearances)
        bottleneck_idx = np.argmin(clearances)
        bottleneck_val = clearances[bottleneck_idx]

        cx, cy = best_centres[bottleneck_idx]
        dist_left = cx + 0.5
        dist_right = 0.5 - cx
        dist_down = cy + 0.5
        dist_up = 0.5 - cy
        min_wall_dist = min(dist_left, dist_right, dist_down, dist_up)

        neighbor_idx = np.argmin(dists[bottleneck_idx])
        neighbor_dist = dists[bottleneck_idx, neighbor_idx]

        if neighbor_dist < 2.0 * min_wall_dist:
            nx, ny = best_centres[neighbor_idx]
            dx = cx - nx
            dy = cy - ny
            norm = np.sqrt(dx * dx + dy * dy)
            if norm > 1e-9:
                dx /= norm
                dy /= norm
            else:
                dx, dy = 1.0, 0.0
            move_x, move_y = dx * step, dy * step
        else:
            if min_wall_dist == dist_left:
                move_x, move_y = step, 0.0
            elif min_wall_dist == dist_right:
                move_x, move_y = -step, 0.0
            elif min_wall_dist == dist_down:
                move_x, move_y = 0.0, step
            else:
                move_x, move_y = 0.0, -step

        new_centres = best_centres.copy()
        new_centres[bottleneck_idx] += [move_x, move_y]
        new_centres[bottleneck_idx] = np.clip(new_centres[bottleneck_idx], -0.5, 0.5)

        new_r = _compute_radius(new_centres)
        if new_r > best_r + 1e-12:
            best_r = new_r
            best_centres = new_centres
            step *= 0.95
        else:
            step *= 0.5

    return best_centres, best_r


def construct():
    start_time = time.time()
    time_budget = TIME_LIMIT - (start_time - START_TIME)

    # 1. Generate initial configuration
    centres, current_r = _generate_hex_init()

    if current_r <= 0:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    # 2. L-BFGS-B optimization with soft repulsive potential
    remaining = time_budget - (time.time() - start_time)
    if remaining > 30:
        centres, current_r = _local_optimize(centres, remaining - 15)

    # 3. Final polish
    remaining = time_budget - (time.time() - start_time)
    if remaining > 5:
        centres, current_r = _delaunay_polish(centres, remaining - 2)

    # 4. Final validation
    final_r = _compute_radius(centres)
    if final_r <= 0:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    return centres
