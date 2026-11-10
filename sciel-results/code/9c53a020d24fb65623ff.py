import numpy as np
import time
import sys
from scipy.spatial.distance import cdist

N = 200
TIME_LIMIT = 140.0  # Safety margin before 150s kill
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
    Generate a high-quality hexagonal packing initialization for N=200.
    Uses a hex grid fitted to the square, selecting exactly N points
    by removing the least central ones, then centers the result.
    """
    best_centres = None
    best_r = -1

    # Try multiple hex spacings; pick the one that yields exactly N points
    # after trimming, or closest to N with good radius.
    for s in np.linspace(0.038, 0.048, 40):
        h_hex = s * np.sqrt(3) / 2.0
        points = []
        num_rows = int(0.5 / h_hex) + 2
        for j in range(-num_rows, num_rows + 1):
            y = j * h_hex
            if abs(y) > 0.5:
                continue
            offset = (j % 2) * s / 2.0
            num_cols = int(0.5 / s) + 2
            for i in range(-num_cols, num_cols + 1):
                x = i * s + offset
                if -0.5 <= x <= 0.5:
                    points.append([x, y])

        points = np.array(points)
        if len(points) < N:
            continue

        # Select N points: prefer central points (smaller |x|,|y|)
        dist_to_center = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        # But we want to keep a connected, well-distributed set.
        # Strategy: sort by distance to center, take the N closest to center,
        # but this may create a blob. Better: take all, then remove the
        # ones furthest from center until we have N.
        order = np.argsort(dist_to_center)
        # Keep the N closest to center
        selected = points[order[:N]]

        # Center the selection
        selected = selected - selected.mean(axis=0)

        # Check bounds
        if np.any(selected < -0.5) or np.any(selected > 0.5):
            continue

        r = _compute_radius(selected)
        if r > best_r:
            best_r = r
            best_centres = selected

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


def _softmin(x, k=50.0):
    """Soft minimum via log-sum-exp of negatives."""
    x = np.asarray(x)
    return -np.logaddexp.reduce(-x / k) * k


def _gradient_step(centres, t, lr, k=30.0):
    """
    One gradient step on the epigraph objective:
    maximize t s.t. t <= d_ij/2 for all i<j, t <= wall_i for all i.
    We minimize L = -t + (1/(2k)) * sum_i (softmin over constraints - t)^2
    Gradient w.r.t. t and centres.
    """
    n = len(centres)
    # Pairwise distances
    diff = centres[:, None, :] - centres[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)

    # Wall distances
    wall = np.min(np.stack([
        0.5 - np.abs(centres[:, 0]),
        0.5 - np.abs(centres[:, 1])
    ], axis=1), axis=1)

    # Constraints: for each point i, the tightest constraint value
    # c_i = min( min_j d_ij/2, wall_i )
    min_pair = np.min(dist, axis=1) / 2.0
    c = np.minimum(min_pair, wall)

    # Soft surrogate for min constraint per point
    # We want gradient of -t + (1/(2k)) * sum_i (softmin_i - t)^2
    # where softmin_i approximates c_i.

    # Compute softmin of (min_pair, wall) per point
    # softmin(a, b) = -k * log(exp(-a/k) + exp(-b/k))
    a = min_pair
    b = wall
    soft_c = -k * np.log(np.exp(-a / k) + np.exp(-b / k))

    # Gradient w.r.t. t: -1 + (1/k) * sum_i (soft_c_i - t)
    grad_t = -1.0 + np.sum(soft_c - t) / k

    # Gradient w.r.t. centres: for each i, d(soft_c_i)/d(centres)
    # soft_c_i = -k * log(exp(-a_i/k) + exp(-b_i/k))
    # Let E = exp(-a_i/k) + exp(-b_i/k)
    # d(soft_c_i)/d(centres) = -k * (1/E) * (dE/d(centres))
    # dE/d(centres) = -exp(-a_i/k)/k * d(a_i)/d(centres) - exp(-b_i/k)/k * d(b_i)/d(centres)
    # = -exp(-a_i/k) * d(a_i)/d(centres) / k - ...
    # So d(soft_c_i)/d(centres) = exp(-a_i/k)/E * d(a_i)/d(centres) + exp(-b_i/k)/E * d(b_i)/d(centres)

    Ea = np.exp(-a / k)
    Eb = np.exp(-b / k)
    E = Ea + Eb
    wa = Ea / E  # weight for pair constraint
    wb = Eb / E  # weight for wall constraint

    # Gradient of a_i = min_j d_ij/2 w.r.t. centres
    # a_i = 0.5 * min_j |c_i - c_j|
    # d(a_i)/d(c_i) = 0.5 * (c_i - c_{j*}) / |c_i - c_{j*}| where j* = argmin_j d_ij
    # d(a_i)/d(c_j) = -0.5 * (c_i - c_{j*}) / |c_i - c_{j*}|

    # For simplicity, use the soft version: weight all neighbors
    # d(soft_a_i)/d(c_i) where soft_a_i = -k * log(sum_j exp(-d_ij/(2k)))
    # This is the "softmin" over j of d_ij/2.
    # Let's compute soft_a_i and its gradient.
    # soft_a_i = -k * log(sum_{j != i} exp(-d_ij / (2k)))
    # d(soft_a_i)/d(c_i) = sum_j exp(-d_ij/(2k)) * d(d_ij/(2k))/d(c_i) / sum_j exp(-d_ij/(2k))
    # d(d_ij)/d(c_i) = (c_i - c_j) / d_ij
    # So d(soft_a_i)/d(c_i) = sum_j w_ij * (c_i - c_j) / (2 d_ij) / sum_j w_ij * ... 
    # Actually: d(soft_a_i)/d(c_i) = [sum_j exp(-d_ij/(2k)) * (c_i - c_j)/(2 d_ij)] / [sum_j exp(-d_ij/(2k))]

    # This is getting complex. Let's use a simpler approach:
    # Just use the hard min for the direction, but smooth the objective value.
    # For the gradient, use the argmin neighbor.

    # Simpler gradient: for each i, if the binding constraint is a pair,
    # push i away from its nearest neighbor. If wall, push away from wall.

    grad_centres = np.zeros_like(centres)

    for i in range(n):
        # Find nearest neighbor
        j_star = np.argmin(dist[i])
        d_ij = dist[i, j_star]
        if d_ij < 1e-9:
            continue
        # Direction from j_star to i
        dir_vec = (centres[i] - centres[j_star]) / d_ij

        # Wall gradient
        wall_grad = np.zeros(2)
        if wall[i] <= min_pair[i] + 1e-9:
            # Wall is binding (or tied)
            if 0.5 - np.abs(centres[i, 0]) <= np.abs(centres[i, 1]) - 0.5 if False else True:
                pass
            # Determine which wall
            dx = centres[i, 0]
            dy = centres[i, 1]
            d_left = dx + 0.5
            d_right = 0.5 - dx
            d_down = dy + 0.5
            d_up = 0.5 - dy
            min_w = min(d_left, d_right, d_down, d_up)
            if min_w == d_left:
                wall_grad = np.array([1.0, 0.0])
            elif min_w == d_right:
                wall_grad = np.array([-1.0, 0.0])
            elif min_w == d_down:
                wall_grad = np.array([0.0, 1.0])
            else:
                wall_grad = np.array([0.0, -1.0])

        # Pair gradient
        pair_grad = dir_vec  # push i away from j_star

        # Combine based on which is tighter
        if min_pair[i] < wall[i]:
            # Pair is binding
            grad_centres[i] += pair_grad
            grad_centres[j_star] -= pair_grad
        else:
            # Wall is binding
            grad_centres[i] += wall_grad

    # Normalize gradient
    norm = np.linalg.norm(grad_centres)
    if norm > 1e-12:
        grad_centres /= norm

    # Update
    new_centres = centres + lr * grad_centres
    new_centres = np.clip(new_centres, -0.5, 0.5)

    # Update t: t = min(c) - small margin
    new_t = np.min(c) - 0.001
    new_t = max(new_t, 0.0)

    return new_centres, new_t, grad_t


def _optimize_repulsive(centres, time_limit):
    """
    Optimize using a repulsive force simulation with a smooth objective.
    Uses gradient descent on the epigraph formulation with adaptive step size.
    """
    current_r = _compute_radius(centres)
    if current_r <= 0:
        return centres, 0.0

    best_centres = centres.copy()
    best_r = current_r
    t = current_r * 0.99  # Start t slightly below current radius

    deadline = time.time() + time_limit
    lr = 0.002
    min_lr = 1e-7
    no_improve = 0
    max_no_improve = 50

    while time.time() < deadline and lr > min_lr:
        # Compute current radius
        r = _compute_radius(best_centres)
        if r > best_r + 1e-10:
            best_r = r
            no_improve = 0
        else:
            no_improve += 1

        if no_improve > max_no_improve:
            lr *= 0.5
            no_improve = 0
            if lr < min_lr:
                break

        # Gradient step
        new_centres, new_t, grad_t = _gradient_step(best_centres, t, lr, k=20.0)

        # Validate
        new_r = _compute_radius(new_centres)
        if new_r > best_r + 1e-12:
            best_r = new_r
            best_centres = new_centres
            t = new_r * 0.99
            lr *= 0.99  # Slightly shrink
        else:
            # Try a smaller step
            half_lr = lr * 0.5
            new_centres2, _, _ = _gradient_step(best_centres, t, half_lr, k=20.0)
            new_r2 = _compute_radius(new_centres2)
            if new_r2 > best_r + 1e-12:
                best_r = new_r2
                best_centres = new_centres2
                t = new_r2 * 0.99
                lr = half_lr
            else:
                lr *= 0.7

    return best_centres, best_r


def _polish(centres, time_limit):
    """
    Final high-precision polish: coordinate descent with very small steps
    on the bottleneck constraints.
    """
    best_centres = centres.copy()
    best_r = _compute_radius(best_centres)
    deadline = time.time() + time_limit
    step = 1e-5

    while time.time() < deadline and step > 1e-9:
        # Find bottleneck
        wall_clearances = np.min(np.stack([
            0.5 - np.abs(best_centres[:, 0]),
            0.5 - np.abs(best_centres[:, 1])
        ], axis=1), axis=1)

        dists = cdist(best_centres, best_centres)
        np.fill_diagonal(dists, np.inf)
        min_neighbor_dists = np.min(dists, axis=1)

        clearances = np.minimum(min_neighbor_dists / 2.0, wall_clearances)
        bottleneck_idx = np.argmin(clearances)
        bottleneck_val = clearances[bottleneck_idx]

        if bottleneck_val > best_r + 1e-10:
            # Try to improve
            pass

        # Determine move direction
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
            step *= 0.98
        else:
            step *= 0.5

    return best_centres, best_r


def construct():
    start_time = time.time()
    time_budget = TIME_LIMIT - (start_time - START_TIME)

    # 1. Generate initial configuration
    centres, current_r = _generate_hex_init()

    if current_r <= 0:
        # Emergency fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    # 2. Repulsive force optimization
    remaining = time_budget - (time.time() - start_time)
    if remaining > 20:
        centres, current_r = _optimize_repulsive(centres, remaining - 10)

    # 3. Final polish
    remaining = time_budget - (time.time() - start_time)
    if remaining > 5:
        centres, current_r = _polish(centres, remaining - 2)

    # 4. Final validation
    final_r = _compute_radius(centres)
    if final_r <= 0:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    return centres
