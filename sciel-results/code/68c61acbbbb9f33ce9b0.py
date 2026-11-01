"""Packing 200 equal circles in a unit square: Multi-temperature soft-min + L-BFGS-B + Active Set Polish."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

N = 200
DEADLINE = 145.0  # seconds; leave margin before 150s kill


def _score(pts):
    """Return the common radius r for a valid configuration, or 0.0 if invalid."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Distance to boundary
    ax = np.abs(pts[:, 0])
    ay = np.abs(pts[:, 1])
    boundary = np.minimum(0.5 - ax, 0.5 - ay)
    min_boundary = np.min(boundary)
    if min_boundary < -1e-10:
        return 0.0
    # Pairwise distances
    dist = cdist(pts, pts)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    r = min(min_boundary, min_dist / 2.0)
    return r if r > 0 else 0.0


def _hex_init(jitter_scale=0.0, seed=0):
    """Generate a hexagonal lattice initialization."""
    best_pts = None
    best_r = 0
    rng = np.random.default_rng(seed)

    # Try a wider range of spacings
    s_est = 1.0 / np.sqrt(N) * 1.2
    for s in np.linspace(s_est * 0.80, s_est * 1.20, 30):
        pts_list = []
        row = 0
        while True:
            y = -0.5 + s / 2 + row * s * np.sqrt(3) / 2
            if y > 0.5 - s / 2:
                break
            offset = (row % 2) * s / 2
            x = -0.5 + s / 2 + offset
            while x <= 0.5 - s / 2:
                pts_list.append((x, y))
                x += s
            row += 1
        if len(pts_list) < N:
            continue
        pts = np.array(pts_list[:N])
        if jitter_scale > 0:
            pts += rng.normal(0, jitter_scale, size=pts.shape)
            pts = np.clip(pts, -0.49, 0.49)
        r = _score(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    if best_pts is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        pts = np.array([(x, y) for y in ticks for x in ticks])
        best_pts = pts[:N].copy()
    return best_pts


def _grid_init(rows, cols, seed=0):
    """Generate a rectangular grid initialization with optional jitter."""
    rng = np.random.default_rng(seed)
    xs = (np.arange(cols) + 0.5) / cols - 0.5
    ys = (np.arange(rows) + 0.5) / rows - 0.5
    pts = np.array([(x, y) for y in ys for x in xs])
    if len(pts) > N:
        pts = pts[:N]
    elif len(pts) < N:
        # Pad with jittered points
        extra = rng.uniform(-0.45, 0.45, size=(N - len(pts), 2))
        pts = np.vstack([pts, extra])
    return pts


def _random_init(seed=0):
    """Generate a random initialization."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-0.45, 0.45, size=(N, 2))
    return pts


def _soft_min_neg(x, alpha):
    """
    Objective function for L-BFGS-B using soft-min surrogate.
    x: flat array of size 2N.
    alpha: sharpness of the soft-min (higher = sharper).
    Returns: -soft_min(distances)
    """
    pts = x.reshape(N, 2)

    # Distances to boundary
    d_x = 0.5 - np.abs(pts[:, 0])
    d_y = 0.5 - np.abs(pts[:, 1])
    boundary_dists = np.minimum(d_x, d_y).flatten()

    # Pairwise distances
    dist = cdist(pts, pts)
    np.fill_diagonal(dist, np.inf)
    pairwise_dists = dist[~np.isinf(dist)]

    # Combine all distances
    all_dists = np.concatenate([boundary_dists, pairwise_dists])

    # Soft min: -1/alpha * log(sum(exp(-alpha * d)))
    # Use log-sum-exp trick for numerical stability
    min_d = np.min(all_dists)
    shifted = -alpha * (all_dists - min_d)
    # Clip to avoid overflow in exp
    shifted = np.clip(shifted, -700, 0)
    val = min_d - (1.0 / alpha) * np.log(np.sum(np.exp(shifted)))

    # We want to maximize min distance, so minimize -min distance.
    return -val


def _optimize_lbfgs_multi_temp(pts, time_budget):
    """
    Run L-BFGS-B optimization with multi-temperature annealing schedule.
    """
    t0 = time.time()
    x0 = pts.flatten()

    # Multi-temperature schedule: start broad, end sharp
    # Higher alpha = sharper approximation to true min
    phases = [
        {"alpha": 10.0, "max_iter": 150, "gtol": 1e-3},
        {"alpha": 30.0, "max_iter": 200, "gtol": 1e-4},
        {"alpha": 80.0, "max_iter": 250, "gtol": 1e-5},
        {"alpha": 200.0, "max_iter": 300, "gtol": 1e-6},
        {"alpha": 500.0, "max_iter": 300, "gtol": 1e-7},
    ]

    for phase in phases:
        elapsed = time.time() - t0
        if elapsed > time_budget:
            break

        alpha = phase["alpha"]

        def obj(x):
            return _soft_min_neg(x, alpha)

        try:
            res = minimize(
                obj, x0, method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={
                    'maxiter': phase["max_iter"],
                    'gtol': phase["gtol"],
                    'maxls': 30,
                }
            )
            x0 = res.x
        except Exception:
            pass

    return x0.reshape(N, 2)


def _force_directed(pts, time_budget):
    """
    Force-directed relaxation: repulsion between points, attraction to center/bounds.
    This helps break symmetries and spread points out quickly.
    """
    t0 = time.time()
    pts = pts.copy()
    current_r = _score(pts)

    # Parameters
    learning_rate = 0.003
    min_lr = 1e-7
    repulsion_power = 2.0
    boundary_repulsion = 0.8

    while time.time() - t0 < time_budget:
        forces = np.zeros_like(pts)

        # Pairwise repulsion
        dists = cdist(pts, pts)
        np.fill_diagonal(dists, np.inf)

        # Avoid division by zero
        mask = dists > 1e-10
        inv_dists = np.zeros_like(dists)
        inv_dists[mask] = 1.0 / (dists[mask] ** repulsion_power)

        # Force direction
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        forces -= np.sum(diff * inv_dists[:, :, np.newaxis], axis=1)

        # Boundary repulsion
        d_x = 0.5 - np.abs(pts[:, 0])
        d_y = 0.5 - np.abs(pts[:, 1])
        boundary_dists = np.minimum(d_x, d_y)

        r_est = current_r
        close_to_boundary = boundary_dists < r_est * 1.1
        if np.any(close_to_boundary):
            center_force = -pts * boundary_repulsion
            forces[close_to_boundary] += center_force[close_to_boundary]

        # Normalize forces to prevent explosion
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        max_norm = 1.0
        forces[force_norms > max_norm] *= (max_norm / force_norms[force_norms > max_norm])

        # Update positions
        pts += learning_rate * forces
        pts = np.clip(pts, -0.5, 0.5)

        new_r = _score(pts)
        if new_r > current_r:
            current_r = new_r
            learning_rate *= 0.995
        else:
            learning_rate *= 0.95

        if learning_rate < min_lr:
            break

    return pts


def _active_set_polish(pts, time_budget):
    """
    High-precision polish: only move points involved in the minimum distance constraints.
    """
    t0 = time.time()
    best_pts = pts.copy()
    best_r = _score(best_pts)

    rng = np.random.default_rng(54321)
    step_size = 0.0005
    min_step = 1e-8

    while time.time() - t0 < time_budget:
        # Compute constraints
        d_x = 0.5 - np.abs(best_pts[:, 0])
        d_y = 0.5 - np.abs(best_pts[:, 1])
        boundary_dists = np.minimum(d_x, d_y)
        min_boundary = np.min(boundary_dists)

        dist = cdist(best_pts, best_pts)
        np.fill_diagonal(dist, np.inf)
        min_pair = np.min(dist)

        current_r = min(min_boundary, min_pair / 2.0)

        # Identify active set: points with distance to boundary < current_r * 1.005
        # and pairs with distance < 2 * current_r * 1.005
        active_idx = set()

        tight_boundary = np.where(boundary_dists < current_r * 1.005)[0]
        active_idx.update(tight_boundary.tolist())

        # Find tight pairs
        tree = cKDTree(best_pts)
        pairs, dists = tree.query(best_pts, k=4)
        for i in range(N):
            for j in range(1, 4):
                if dists[i, j] < 2.0 * current_r * 1.005:
                    active_idx.add(i)
                    active_idx.add(int(pairs[i, j]))

        if not active_idx:
            break

        active_list = list(active_idx)
        n_active = len(active_list)

        # Try perturbing active points with decreasing step size
        improved = False
        for attempt in range(20):
            perturb = np.zeros((N, 2))
            perturb[active_list] = rng.normal(0, step_size, size=(n_active, 2))
            trial = best_pts + perturb
            trial = np.clip(trial, -0.5, 0.5)

            trial_r = _score(trial)
            if trial_r > best_r + 1e-12:
                best_pts = trial
                best_r = trial_r
                improved = True
                break

        if improved:
            step_size *= 0.95
        else:
            step_size *= 0.7
            if step_size < min_step:
                break

    return best_pts


def _high_precision_polish(pts, time_budget):
    """
    Ultra-high precision polish using very sharp soft-min with L-BFGS-B.
    """
    t0 = time.time()
    x0 = pts.flatten()

    # Very sharp temperatures for final precision
    phases = [
        {"alpha": 1000.0, "max_iter": 500, "gtol": 1e-9},
        {"alpha": 2000.0, "max_iter": 500, "gtol": 1e-10},
        {"alpha": 5000.0, "max_iter": 500, "gtol": 1e-11},
    ]

    for phase in phases:
        elapsed = time.time() - t0
        if elapsed > time_budget:
            break

        alpha = phase["alpha"]

        def obj(x):
            return _soft_min_neg(x, alpha)

        try:
            res = minimize(
                obj, x0, method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={
                    'maxiter': phase["max_iter"],
                    'gtol': phase["gtol"],
                    'maxls': 50,
                }
            )
            x0 = res.x
        except Exception:
            pass

    return x0.reshape(N, 2)


def construct():
    t_start = time.time()

    # 1. Initialization: try multiple diverse inits
    best_pts = None
    best_r = 0.0

    inits = [
        _hex_init(jitter_scale=0.0, seed=0),
        _hex_init(jitter_scale=0.003, seed=1),
        _hex_init(jitter_scale=0.008, seed=2),
        _hex_init(jitter_scale=0.015, seed=3),
        _grid_init(14, 15, seed=4),
        _grid_init(15, 14, seed=5),
        _random_init(seed=6),
        _random_init(seed=7),
    ]

    for pts in inits:
        r = _score(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    # 2. Force-Directed Relaxation
    remaining = DEADLINE - time.time()
    if remaining > 15:
        fd_budget = min(remaining * 0.25, 15.0)
        fd_pts = _force_directed(best_pts, fd_budget)
        fd_r = _score(fd_pts)
        if fd_r > best_r:
            best_r = fd_r
            best_pts = fd_pts.copy()

    # 3. Multi-temperature L-BFGS-B Polish
    remaining = DEADLINE - time.time()
    if remaining > 20:
        lbfgs_budget = min(remaining * 0.40, 40.0)
        opt_pts = _optimize_lbfgs_multi_temp(best_pts, lbfgs_budget)
        opt_pts = np.clip(opt_pts, -0.5, 0.5)
        opt_r = _score(opt_pts)
        if opt_r > best_r:
            best_r = opt_r
            best_pts = opt_pts.copy()

    # 4. Active Set Polish
    remaining = DEADLINE - time.time()
    if remaining > 8:
        polish_budget = min(remaining * 0.30, 20.0)
        polished_pts = _active_set_polish(best_pts, polish_budget)
        polished_r = _score(polished_pts)
        if polished_r > best_r:
            best_r = polished_r
            best_pts = polished_pts.copy()

    # 5. High-precision L-BFGS-B pass
    remaining = DEADLINE - time.time()
    if remaining > 15:
        hp_budget = min(remaining * 0.35, 25.0)
        hp_pts = _high_precision_polish(best_pts, hp_budget)
        hp_pts = np.clip(hp_pts, -0.5, 0.5)
        hp_r = _score(hp_pts)
        if hp_r > best_r:
            best_r = hp_r
            best_pts = hp_pts.copy()

    # 6. Final Active Set Polish
    remaining = DEADLINE - time.time()
    if remaining > 5:
        polish_budget = min(remaining * 0.60, 10.0)
        polished_pts = _active_set_polish(best_pts, polish_budget)
        polished_r = _score(polished_pts)
        if polished_r > best_r:
            best_r = polished_r
            best_pts = polished_pts.copy()

    # 7. Final high-precision pass if time remains
    remaining = DEADLINE - time.time()
    if remaining > 8:
        hp_budget = min(remaining * 0.70, 8.0)
        hp_pts = _high_precision_polish(best_pts, hp_budget)
        hp_pts = np.clip(hp_pts, -0.5, 0.5)
        hp_r = _score(hp_pts)
        if hp_r > best_r:
            best_r = hp_r
            best_pts = hp_pts.copy()

    if best_pts is None:
        best_pts = _hex_init()

    return best_pts
