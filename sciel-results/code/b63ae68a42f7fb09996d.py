"""
Fifty equal circles in a unit square - constraint satisfaction approach.
Uses projected gradient descent with explicit collision/wall constraints,
binary search on target radius, and multi-restart strategy.
"""
import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import cKDTree

N = 50
DEADLINE = 145.0  # Safety margin before 150s kill


def _compute_radius(centers):
    """Compute the true maximum radius for a given configuration."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0

    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wall_dists_x, wall_dists_y))

    if min_wall < 0:
        return 0.0

    tree = cKDTree(centers)
    dists, idx = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1]) / 2.0

    return min(min_pair, min_wall)


def _project_constraints(centers, r):
    """
    Project centers to satisfy: pairwise dist >= 2r, wall dist >= r.
    Uses iterative pairwise separation and wall clipping.
    """
    centers = centers.copy()
    for _ in range(30):
        # Wall clipping
        centers[:, 0] = np.clip(centers[:, 0], -0.5 + r, 0.5 - r)
        centers[:, 1] = np.clip(centers[:, 1], -0.5 + r, 0.5 - r)

        # Pairwise separation via cKDTree
        tree = cKDTree(centers)
        dists, idx = tree.query(centers, k=2)
        for i in range(N):
            j = idx[i, 1]
            d = dists[i, 1]
            if d < 2 * r - 1e-12:
                if d < 1e-12:
                    # Random separation
                    angle = np.random.uniform(0, 2 * np.pi)
                    offset = np.array([np.cos(angle), np.sin(angle)]) * (r - d / 2 + 1e-6)
                    centers[i] += offset
                    centers[j] -= offset
                else:
                    diff = centers[i] - centers[j]
                    unit = diff / d
                    needed = (2 * r - d) / 2.0
                    centers[i] += unit * needed
                    centers[j] -= unit * needed

        # Re-clip walls after separation
        centers[:, 0] = np.clip(centers[:, 0], -0.5 + r, 0.5 - r)
        centers[:, 1] = np.clip(centers[:, 1], -0.5 + r, 0.5 - r)

    return centers


def _violation(centers, r):
    """Compute total constraint violation for target radius r."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return np.inf

    # Wall violations
    wall_x = np.maximum(0, r - (0.5 - np.abs(centers[:, 0])))
    wall_y = np.maximum(0, r - (0.5 - np.abs(centers[:, 1])))
    wall_viol = np.sum(wall_x ** 2) + np.sum(wall_y ** 2)

    # Pairwise violations using cKDTree
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    pair_viol = np.sum(np.maximum(0, 2 * r - dists[:, 1]) ** 2)

    return wall_viol + pair_viol


def _violation_grad(centers, r):
    """Compute gradient of violation w.r.t. centers."""
    centers = np.asarray(centers, dtype=np.float64)
    grad = np.zeros_like(centers)

    # Wall gradients
    wx = r - (0.5 - np.abs(centers[:, 0]))
    wy = r - (0.5 - np.abs(centers[:, 1]))
    mask_x = wx > 0
    mask_y = wy > 0
    grad[mask_x, 0] += 2 * wx[mask_x] * np.sign(centers[mask_x, 0])
    grad[mask_y, 1] += 2 * wy[mask_y] * np.sign(centers[mask_y, 1])

    # Pairwise gradients
    tree = cKDTree(centers)
    dists, idx = tree.query(centers, k=2)
    for i in range(N):
        j = idx[i, 1]
        d = dists[i, 1]
        if d < 2 * r:
            if d < 1e-12:
                continue
            diff = centers[i] - centers[j]
            factor = 2 * (2 * r - d) / d
            grad[i] -= factor * diff
            grad[j] += factor * diff

    return grad


def _projected_gradient_descent(centers, r, max_iter=200, lr=None, time_limit=None):
    """
    Projected gradient descent to minimize violation for target radius r.
    Returns (centers, final_violation).
    """
    start = time.time()
    centers = centers.copy()
    if lr is None:
        lr = 0.002

    for it in range(max_iter):
        if time_limit is not None and time.time() - start > time_limit:
            break

        # Project to feasible region (walls at minimum)
        centers[:, 0] = np.clip(centers[:, 0], -0.5 + r, 0.5 - r)
        centers[:, 1] = np.clip(centers[:, 1], -0.5 + r, 0.5 - r)

        viol = _violation(centers, r)
        if viol < 1e-14:
            break

        grad = _violation_grad(centers, r)
        centers -= lr * grad

        # Re-project
        centers[:, 0] = np.clip(centers[:, 0], -0.5 + r, 0.5 - r)
        centers[:, 1] = np.clip(centers[:, 1], -0.5 + r, 0.5 - r)

    final_viol = _violation(centers, r)
    return centers, final_viol


def _feasibility_check(centers, r, time_budget=5.0):
    """
    Check if radius r is feasible by running projected GD from multiple starts.
    Returns (best_centers, best_violation, feasible).
    """
    start_time = time.time()
    best_centers = centers.copy()
    best_viol = _violation(centers, r)
    feasible = best_viol < 1e-10

    # Try multiple restarts with different learning rates
    rng = np.random.default_rng(42)
    for restart in range(5):
        if time.time() - start_time > time_budget:
            break

        c = centers.copy()
        # Add small perturbation for restarts > 0
        if restart > 0:
            c += rng.normal(0, 0.005, c.shape)
            c[:, 0] = np.clip(c[:, 0], -0.5 + r, 0.5 - r)
            c[:, 1] = np.clip(c[:, 1], -0.5 + r, 0.5 - r)

        lr = 0.003 / (1 + 0.5 * restart)
        c, viol = _projected_gradient_descent(
            c, r, max_iter=150, lr=lr,
            time_limit=min(1.0, time_budget - (time.time() - start_time))
        )

        if viol < best_viol:
            best_viol = viol
            best_centers = c
            if viol < 1e-10:
                feasible = True
                break

    return best_centers, best_viol, feasible


def _binary_search_radius(centers, r_lo, r_hi, time_budget):
    """
    Binary search for maximum feasible radius.
    """
    start_time = time.time()
    best_centers = centers.copy()
    best_r = r_lo

    for _ in range(20):
        if time.time() - start_time > time_budget:
            break

        mid = (r_lo + r_hi) / 2
        remaining = time_budget - (time.time() - start_time)
        if remaining < 2.0:
            break

        c, viol, feasible = _feasibility_check(
            best_centers, mid, time_budget=min(remaining * 0.6, 6.0)
        )

        if feasible:
            r_lo = mid
            best_r = mid
            best_centers = c
        else:
            r_hi = mid

    return best_centers, best_r


def _initial_hexagonal():
    """Create initial hexagonal packing layout."""
    row_counts = [8, 7, 8, 7, 8, 7, 5]
    points = []
    total_rows = len(row_counts)
    vert_spacing = 0.9 / (total_rows - 1)

    for row_idx, count in enumerate(row_counts):
        y = -0.45 + row_idx * vert_spacing
        horiz_spacing = 0.9 / count if count > 1 else 0
        x_start = -0.45 + (0.9 - horiz_spacing * (count - 1)) / 2
        if row_idx % 2 == 1:
            x_offset = horiz_spacing / 2
        else:
            x_offset = 0
        for i in range(count):
            x = x_start + i * horiz_spacing + x_offset
            points.append([x, y])

    points = np.array(points[:N])
    rng = np.random.default_rng(12345)
    points = points + rng.normal(0, 0.005, points.shape)
    points = np.clip(points, -0.49, 0.49)
    return points


def _initial_grid():
    """Create initial grid layout with jitter."""
    # 7x7 grid with 2 removed = 49, add 1 more
    # Actually use a different arrangement
    row_counts = [8, 7, 8, 7, 8, 7, 5]
    points = []
    total_rows = len(row_counts)
    vert_spacing = 0.9 / (total_rows - 1)

    for row_idx, count in enumerate(row_counts):
        y = -0.45 + row_idx * vert_spacing
        horiz_spacing = 0.9 / count if count > 1 else 0
        x_start = -0.45 + (0.9 - horiz_spacing * (count - 1)) / 2
        if row_idx % 2 == 1:
            x_offset = horiz_spacing / 2
        else:
            x_offset = 0
        for i in range(count):
            x = x_start + i * horiz_spacing + x_offset
            points.append([x, y])

    points = np.array(points[:N])
    rng = np.random.default_rng(999)
    points = points + rng.normal(0, 0.01, points.shape)
    points = np.clip(points, -0.49, 0.49)
    return points


def _simulated_annealing(centers, r_target, rng, time_budget):
    """
    Simulated annealing for the constraint satisfaction problem.
    Moves circles to reduce violation for target radius r_target.
    """
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_viol = _violation(current, r_target)
    best_r = _compute_radius(current)

    T_start = 0.01
    T_end = 1e-8
    n_steps = 50000
    cooling = (T_end / T_start) ** (1.0 / max(n_steps, 1))
    T = T_start

    for step in range(n_steps):
        if time.time() - start_time > time_budget:
            break

        curr_viol = _violation(current, r_target)

        # Pick a circle to move, biased toward those with high violation
        tree = cKDTree(current)
        dists, _ = tree.query(current, k=2)
        wall_x = 0.5 - np.abs(current[:, 0])
        wall_y = 0.5 - np.abs(current[:, 1])

        # Compute per-circle violation contribution
        circle_viol = np.zeros(N)
        for i in range(N):
            j = tree.query(current[i], k=2)[1][1]
            d = np.linalg.norm(current[i] - current[j])
            circle_viol[i] += np.maximum(0, 2 * r_target - d) ** 2
            circle_viol[j] += np.maximum(0, 2 * r_target - d) ** 2
            circle_viol[i] += np.maximum(0, r_target - wall_x[i]) ** 2
            circle_viol[i] += np.maximum(0, r_target - wall_y[i]) ** 2

        # Normalize for probability
        total = np.sum(circle_viol)
        if total < 1e-15:
            probs = np.ones(N) / N
        else:
            probs = circle_viol / total

        i = rng.choice(N, p=probs)

        # Random move
        move_size = 0.003 + 0.015 * T / T_start
        delta = rng.normal(0, move_size, 2)
        trial = current.copy()
        trial[i] += delta
        trial[i, 0] = np.clip(trial[i, 0], -0.5 + r_target, 0.5 - r_target)
        trial[i, 1] = np.clip(trial[i, 1], -0.5 + r_target, 0.5 - r_target)

        trial_viol = _violation(trial, r_target)
        delta_viol = trial_viol - curr_viol

        if delta_viol <= 0:
            current = trial
            if trial_viol < best_viol:
                best_viol = trial_viol
                best = trial.copy()
        else:
            if rng.random() < np.exp(-delta_viol / max(T, 1e-12)):
                current = trial

        T *= cooling

    # Update best radius
    best_r = _compute_radius(best)
    return best, best_r


def _lbfgs_polish(centers, time_budget):
    """
    L-BFGS polish using soft-min objective for final refinement.
    """
    start_time = time.time()

    def _soft_min_obj(x, tau=0.005):
        c = x.reshape(-1, 2)
        wall_x = 0.5 - np.abs(c[:, 0])
        wall_y = 0.5 - np.abs(c[:, 1])
        walls = np.minimum(wall_x, wall_y)

        tree = cKDTree(c)
        dists, _ = tree.query(c, k=2)
        pair_dists = dists[:, 1] / 2.0

        all_dists = np.concatenate([walls, pair_dists])
        min_d = np.min(all_dists)
        shifted = all_dists - min_d
        log_sum = np.log(np.sum(np.exp(-shifted / tau)))
        soft_min = min_d - tau * log_sum
        return -soft_min

    def _soft_min_grad(x, tau=0.005):
        c = x.reshape(-1, 2)
        wall_x = 0.5 - np.abs(c[:, 0])
        wall_y = 0.5 - np.abs(c[:, 1])
        walls = np.minimum(wall_x, wall_y)

        tree = cKDTree(c)
        dists, idx = tree.query(c, k=2)
        pair_dists = dists[:, 1] / 2.0

        all_dists = np.concatenate([walls, pair_dists])
        min_d = np.min(all_dists)
        shifted = all_dists - min_d
        exp_vals = np.exp(-shifted / tau)
        weights = exp_vals / np.sum(exp_vals)

        grad = np.zeros_like(x)

        for i in range(N):
            w = weights[i]
            if walls[i] == wall_x[i]:
                grad[2 * i] += w * np.sign(c[i, 0])
            if walls[i] == wall_y[i]:
                grad[2 * i + 1] += w * np.sign(c[i, 1])

        for i in range(N):
            w = weights[N + i]
            if w < 1e-15:
                continue
            j = idx[i, 1]
            d = dists[i, 1]
            if d > 1e-12:
                factor = 0.5 * w / d
                diff = c[i] - c[j]
                grad[2 * i:2 * i + 2] -= factor * diff
                grad[2 * j:2 * j + 2] += factor * diff

        return grad

    x0 = centers.flatten().copy()

    try:
        result = opt.minimize(
            _soft_min_obj, x0, jac=_soft_min_grad, method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10,
                     'maxfun': 500, 'disp': False}
        )
        polished = result.x.reshape(-1, 2)
        polished = np.clip(polished, -0.5, 0.5)
        return polished
    except Exception:
        return centers


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    # Initial configuration
    best_centers = _initial_hexagonal()
    best_radius = _compute_radius(best_centers)

    rng = np.random.default_rng(777)

    # Phase 1: Simulated annealing to find good initial configuration
    sa_budget = min(40.0, (deadline - time.time()) * 0.3)
    if sa_budget > 5:
        # Use current best radius as target for SA
        r_target = best_radius * 1.01
        best_centers, sa_r = _simulated_annealing(best_centers, r_target, rng, sa_budget)
        if sa_r > best_radius:
            best_radius = sa_r

    # Phase 2: L-BFGS polish
    if time.time() < deadline - 40:
        trial = _lbfgs_polish(best_centers, time_budget=15.0)
        trial_r = _compute_radius(trial)
        if trial_r > best_radius:
            best_radius = trial_r
            best_centers = trial

    # Phase 3: Binary search on radius with constraint satisfaction
    if time.time() < deadline - 20:
        r_lo = best_radius
        r_hi = min(0.075, best_radius * 1.05)
        remaining = deadline - time.time() - 10
        best_centers, best_radius = _binary_search_radius(
            best_centers, r_lo, r_hi, time_budget=remaining
        )

    # Phase 4: Final L-BFGS polish
    if time.time() < deadline - 10:
        trial = _lbfgs_polish(best_centers, time_budget=8.0)
        trial_r = _compute_radius(trial)
        if trial_r > best_radius:
            best_radius = trial_r
            best_centers = trial

    # Final validation
    if not np.all(np.isfinite(best_centers)) or best_centers.shape != (N, 2):
        return _initial_hexagonal()

    return best_centers
