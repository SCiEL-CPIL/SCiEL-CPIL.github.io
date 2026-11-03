"""
Pack 200 equal circles in a unit square using gradient-based optimization
with a soft-min objective, momentum, and SLSQP polish.
"""
import numpy as np
import time
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

N = 200
_TIME_LIMIT = 135  # Leave margin for 150s kill


def _compute_radius_fast(pts):
    """
    Fast radius computation using cdist.
    """
    if len(pts) == 0 or not np.all(np.isfinite(pts)):
        return 0.0

    # Wall distances
    min_wall = np.min([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1])
    ])

    if min_wall < 0:
        return 0.0

    # Pairwise distances
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D) / 2.0

    return min(min_wall, min_pair)


def _soft_min_value_and_grad(pts, beta=50.0):
    """
    Compute soft-min of all pairwise distances (scaled) and wall distances,
    plus gradient w.r.t. points.
    """
    n = len(pts)
    x = pts[:, 0]
    y = pts[:, 1]

    # Pairwise distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    D = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(D, np.inf)

    # Wall distances (4 per point)
    w_left = x + 0.5
    w_right = 0.5 - x
    w_bottom = y + 0.5
    w_top = 0.5 - y

    # Pair effective: D/2
    pair_eff = D / 2.0
    # Wall effective: the 4 wall distances
    wall_eff = np.column_stack([w_left, w_right, w_bottom, w_top])  # (n, 4)

    # Flatten all values
    all_vals = np.concatenate([pair_eff.ravel(), wall_eff.ravel()])
    # Remove inf (diagonal)
    all_vals = all_vals[np.isfinite(all_vals)]

    if len(all_vals) == 0:
        return 0.0, np.zeros_like(pts)

    # Soft-min: -logsumexp(-beta * vals) / beta
    shifted = -beta * all_vals
    max_shifted = np.max(shifted)
    logsumexp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    soft_min_val = -logsumexp / beta

    # Gradient w.r.t. all_vals: softmax weights
    weights = np.exp(shifted - max_shifted)
    weights /= np.sum(weights)

    # Build gradient
    grad = np.zeros_like(pts)

    # Pair part
    pair_weights = weights[:pair_eff.size].reshape(D.shape)
    safe_D = np.where(D > 1e-12, D, 1.0)
    factor = pair_weights / (2.0 * safe_D)
    factor = np.where(np.isfinite(factor), factor, 0.0)

    gx = np.sum(factor * dx, axis=1)
    gy = np.sum(factor * dy, axis=1)
    grad[:, 0] += gx
    grad[:, 1] += gy

    # Wall part
    wall_weights = weights[pair_eff.size:]
    ww = wall_weights.reshape(n, 4)
    grad[:, 0] += ww[:, 0] - ww[:, 1]
    grad[:, 1] += ww[:, 2] - ww[:, 3]

    return soft_min_val, grad


def _project_into_square(pts):
    """Clip points to be inside the square."""
    return np.clip(pts, -0.5, 0.5)


def _local_optimize(coords, max_iter=3000, time_limit=None, start_time=None):
    """
    Gradient-based optimization using soft-min surrogate with momentum.
    """
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = _TIME_LIMIT - 10

    coords = coords.copy()
    n = len(coords)

    beta_start = 30.0
    beta_end = 200.0

    velocity = np.zeros_like(coords)
    momentum = 0.3
    lr = 0.002

    best_r = _compute_radius_fast(coords)
    best_coords = coords.copy()
    no_improve = 0
    max_no_improve = 80

    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break

        frac = min(1.0, it / max_iter)
        beta = beta_start + (beta_end - beta_start) * frac**1.5

        val, grad = _soft_min_value_and_grad(coords, beta=beta)

        step = lr * grad
        coords_new = coords + momentum * velocity + step
        coords_new = _project_into_square(coords_new)
        velocity = momentum * velocity + step

        r = _compute_radius_fast(coords_new)

        if r > best_r + 1e-12:
            best_r = r
            best_coords = coords_new.copy()
            no_improve = 0
            coords = coords_new
        else:
            no_improve += 1
            if no_improve % 20 == 0:
                lr *= 0.85
            if no_improve >= max_no_improve and it % 50 == 0:
                perturb = 0.002 * (1.0 / beta + 0.01) * np.random.randn(n, 2)
                coords_new = coords + perturb
                coords_new = _project_into_square(coords_new)
                r_pert = _compute_radius_fast(coords_new)
                if r_pert > best_r:
                    best_r = r_pert
                    best_coords = coords_new.copy()
                    coords = coords_new
                    no_improve = 0
            if no_improve > max_no_improve * 3:
                kick = 0.01 * np.random.randn(n, 2)
                coords_kick = coords + kick
                coords_kick = _project_into_square(coords_kick)
                r_kick = _compute_radius_fast(coords_kick)
                if r_kick > best_r:
                    best_r = r_kick
                    best_coords = coords_kick.copy()
                    coords = coords_kick
                    no_improve = 0
                    lr = 0.002

    return best_coords


def _slsqp_polish(coords, time_limit=10.0, start_time=None):
    """
    High-precision polish using scipy SLSQP with explicit constraints.
    Maximizes radius r subject to non-overlap and boundary constraints.
    """
    if start_time is None:
        start_time = time.time()

    coords = coords.copy()
    n = len(coords)

    # Compute current radius
    current_r = _compute_radius_fast(coords)

    # Precompute pairs for efficiency
    # We'll use a subset of the most critical constraints to keep SLSQP fast
    # Find the k closest pairs
    D = cdist(coords, coords)
    np.fill_diagonal(D, np.inf)

    # Get indices of closest pairs
    D_flat = D.ravel()
    k_closest = min(500, D_flat.size)
    idx = np.argpartition(D_flat, k_closest)[:k_closest]
    i_idx, j_idx = np.unravel_index(idx, D.shape)

    # Filter unique pairs (i < j)
    unique_mask = i_idx < j_idx
    i_idx = i_idx[unique_mask]
    j_idx = j_idx[unique_mask]

    # Also add boundary constraints for points near walls
    # Points within 2*r of any wall
    wall_margin = 2 * current_r
    near_left = coords[:, 0] < -0.5 + wall_margin
    near_right = coords[:, 0] > 0.5 - wall_margin
    near_bottom = coords[:, 1] < -0.5 + wall_margin
    near_top = coords[:, 1] > 0.5 - wall_margin

    def objective(x):
        """Minimize negative radius (maximize radius)."""
        pts = x.reshape(-1, 2)
        # Compute radius
        min_wall = np.min([
            np.min(pts[:, 0]) + 0.5,
            0.5 - np.max(pts[:, 0]),
            np.min(pts[:, 1]) + 0.5,
            0.5 - np.max(pts[:, 1])
        ])
        # Check closest pairs
        if len(i_idx) > 0:
            d_ij = np.sqrt(np.sum((pts[i_idx] - pts[j_idx])**2, axis=1))
            min_pair = np.min(d_ij) / 2.0
        else:
            min_pair = min_wall

        return -min(min_wall, min_pair)

    def objective_grad(x):
        """Gradient of objective."""
        pts = x.reshape(-1, 2)
        grad = np.zeros_like(x)

        # Wall gradients
        min_wall = np.min([
            np.min(pts[:, 0]) + 0.5,
            0.5 - np.max(pts[:, 0]),
            np.min(pts[:, 1]) + 0.5,
            0.5 - np.max(pts[:, 1])
        ])

        # Determine which wall is active
        wl = np.min(pts[:, 0]) + 0.5
        wr = 0.5 - np.max(pts[:, 0])
        wb = np.min(pts[:, 1]) + 0.5
        wt = 0.5 - np.max(pts[:, 1])

        # Pair gradients
        if len(i_idx) > 0:
            d_ij = np.sqrt(np.sum((pts[i_idx] - pts[j_idx])**2, axis=1))
            min_pair = np.min(d_ij) / 2.0
        else:
            min_pair = min_wall

        r = min(min_wall, min_pair)

        # Gradient contribution from active constraint
        if min_wall <= min_pair:
            # Wall is active
            if wl == min_wall:
                idx_min = np.argmin(pts[:, 0])
                grad[idx_min * 2] = 1.0
            elif wr == min_wall:
                idx_max = np.argmax(pts[:, 0])
                grad[idx_max * 2] = -1.0
            elif wb == min_wall:
                idx_min = np.argmin(pts[:, 1])
                grad[idx_min * 2 + 1] = 1.0
            elif wt == min_wall:
                idx_max = np.argmax(pts[:, 1])
                grad[idx_max * 2 + 1] = -1.0
        else:
            # Pair is active
            if len(i_idx) > 0:
                d_ij = np.sqrt(np.sum((pts[i_idx] - pts[j_idx])**2, axis=1))
                min_idx = np.argmin(d_ij)
                i = i_idx[min_idx]
                j = j_idx[min_idx]
                diff = pts[i] - pts[j]
                dist = d_ij[min_idx]
                if dist > 1e-12:
                    grad[i * 2] = diff[0] / dist
                    grad[i * 2 + 1] = diff[1] / dist
                    grad[j * 2] = -diff[0] / dist
                    grad[j * 2 + 1] = -diff[1] / dist

        # Negate because we minimize -r
        return -grad

    # Define constraints
    constraints = []

    # Non-overlap constraints for closest pairs: d_ij >= 2*r
    # Since r is variable, we use: d_ij^2 - (2*r)^2 >= 0
    # But r is not a variable here; we fix r and check feasibility
    # Actually, for SLSQP we need constraints in terms of x only.
    # We'll use: d_ij >= 2 * current_r (conservative)
    # Or better: use the soft constraint approach

    # Boundary constraints: points must be in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (2 * n)

    # Try SLSQP
    try:
        result = minimize(
            objective,
            coords.ravel(),
            jac=objective_grad,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-10, 'disp': False}
        )

        if result.success or result.fun < objective(coords.ravel()):
            new_coords = result.x.reshape(-1, 2)
            new_coords = np.clip(new_coords, -0.5, 0.5)
            return new_coords
    except Exception:
        pass

    return coords


def _high_precision_polish(coords, time_limit=10.0, start_time=None):
    """
    Final high-precision polish: combine soft-min gradient and SLSQP.
    """
    if start_time is None:
        start_time = time.time()

    coords = coords.copy()
    n = len(coords)

    velocity = np.zeros_like(coords)
    momentum = 0.2
    lr = 0.0005

    best_r = _compute_radius_fast(coords)
    best_coords = coords.copy()

    max_iter = 3000
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break

        beta = 300.0
        val, grad = _soft_min_value_and_grad(coords, beta=beta)

        step = lr * grad
        coords_new = coords + momentum * velocity + step
        coords_new = _project_into_square(coords_new)
        velocity = momentum * velocity + step

        r = _compute_radius_fast(coords_new)
        if r > best_r + 1e-14:
            best_r = r
            best_coords = coords_new.copy()
            coords = coords_new
        else:
            lr *= 0.98
            if lr < 1e-8:
                break

    # Try SLSQP polish on the best result
    remaining = time_limit - (time.time() - start_time)
    if remaining > 2:
        try:
            slsqp_result = _slsqp_polish(best_coords, time_limit=remaining,
                                         start_time=start_time)
            r_slsqp = _compute_radius_fast(slsqp_result)
            if r_slsqp > best_r:
                best_r = r_slsqp
                best_coords = slsqp_result
        except Exception:
            pass

    return best_coords


def _hexagonal_initialization(n, jitter=0.0):
    """
    Generate a hexagonal layout with optional jitter.
    """
    m = int(np.ceil(np.sqrt(n)))
    r_est = 0.5 / m

    cols = int(1.0 / (2 * r_est)) + 2
    rows = int(1.0 / (r_est * np.sqrt(3))) + 2

    points = []
    for row in range(rows):
        y = -0.5 + r_est + row * r_est * np.sqrt(3)
        offset = 0.0 if row % 2 == 0 else r_est
        for col in range(cols):
            x = -0.5 + r_est + offset + col * 2 * r_est
            points.append([x, y])

    points = np.array(points)

    mask = (points[:, 0] >= -0.5) & (points[:, 0] <= 0.5) & (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
    points = points[mask]

    if len(points) < n:
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        points = grid[:n]
    else:
        dists_to_center = np.sum(points**2, axis=1)
        indices = np.argsort(dists_to_center)
        points = points[indices[:n]]

    if jitter > 0:
        points += jitter * np.random.randn(n, 2)
        points = np.clip(points, -0.5, 0.5)

    return points


def _grid_initialization(n, jitter=0.0):
    """
    Generate a grid layout with optional jitter.
    """
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])

    if len(grid) < n:
        extra = n - len(grid)
        extra_pts = np.random.uniform(-0.45, 0.45, (extra, 2))
        grid = np.vstack([grid, extra_pts])

    grid = grid[:n]

    if jitter > 0:
        grid += jitter * np.random.randn(n, 2)
        grid = np.clip(grid, -0.5, 0.5)

    return grid


def _random_initialization(n):
    """
    Generate random points.
    """
    return np.random.uniform(-0.45, 0.45, (n, 2))


def construct():
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT

    initializers = [
        ("hex", lambda: _hexagonal_initialization(N, jitter=0.0)),
        ("hex_jitter", lambda: _hexagonal_initialization(N, jitter=0.015)),
        ("grid", lambda: _grid_initialization(N, jitter=0.0)),
        ("grid_jitter", lambda: _grid_initialization(N, jitter=0.015)),
        ("random", lambda: _random_initialization(N)),
        ("hex_jitter2", lambda: _hexagonal_initialization(N, jitter=0.03)),
    ]

    best_coords = None
    best_radius = 0.0

    # Phase 1: Run gradient optimization on multiple starts
    for name, init_func in initializers:
        if time.time() > deadline - 40:
            break

        try:
            coords = init_func()
            r = _compute_radius_fast(coords)

            if r > best_radius:
                best_radius = r
                best_coords = coords.copy()

            remaining_time = deadline - time.time() - 30
            if remaining_time > 8:
                optimized = _local_optimize(coords, max_iter=2000,
                                           time_limit=remaining_time,
                                           start_time=start_time)
                r_opt = _compute_radius_fast(optimized)

                if r_opt > best_radius:
                    best_radius = r_opt
                    best_coords = optimized.copy()
        except Exception:
            continue

    # Phase 2: High-precision polish on best result (soft-min + SLSQP)
    if best_coords is not None:
        remaining_time = deadline - time.time() - 5
        if remaining_time > 3:
            try:
                polished = _high_precision_polish(best_coords,
                                                  time_limit=remaining_time,
                                                  start_time=start_time)
                r_pol = _compute_radius_fast(polished)
                if r_pol > best_radius:
                    best_radius = r_pol
                    best_coords = polished
            except Exception:
                pass

    # Final validation
    if best_coords is None or not np.all(np.isfinite(best_coords)):
        return _grid_initialization(N)

    best_coords = np.clip(best_coords, -0.4999, 0.4999)

    return best_coords


# EVOLVE-BLOCK-START
def compute_radius(coords):
    return _compute_radius_fast(coords)


def objective(coords):
    r = compute_radius(coords)
    return -r


def initial_hex_layout(n):
    return _hexagonal_initialization(n)
# EVOLVE-BLOCK-END
