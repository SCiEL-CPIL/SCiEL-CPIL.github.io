"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 150.0
SAFETY_MARGIN = 10.0


def _compute_score(points):
    """Compute the score: min of half-min-pair-distance and min distance to boundary."""
    n = len(points)
    if n == 0:
        return 0.0

    # Min distance to boundary
    dists_to_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = float(np.min(dists_to_boundary))

    # Min pair distance via vectorized computation
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair = float(np.min(dist_matrix))

    return min(min_pair / 2.0, min_boundary)


def _compute_score_fast(points, cache=None):
    """Compute score with optional cache. Returns (score, dist_matrix, boundary_dists)."""
    n = len(points)
    if n == 0:
        return 0.0, None, None

    # Min distance to boundary
    norms = np.linalg.norm(points, axis=1)
    dists_to_boundary = 1.0 - norms
    min_boundary = float(np.min(dists_to_boundary))

    # Min pair distance
    if cache is not None and cache.shape == (n, n):
        dist_matrix = cache
    else:
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist_matrix, np.inf)
    min_pair = float(np.min(dist_matrix))

    return min(min_pair / 2.0, min_boundary), dist_matrix, dists_to_boundary


def _soft_min_gradient(pos, t=80.0):
    """
    Compute gradient of the soft-min of all pairwise and boundary distances.
    We want to maximize the minimum distance, so we minimize the soft-min.
    Returns (gradient, soft_min_value, actual_min).
    """
    n = len(pos)
    pos = pos.reshape(n, 2)

    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)

    # Boundary distances
    norms = np.linalg.norm(pos, axis=1)
    boundary_dists = 1.0 - norms

    # Flatten pairwise distances (upper triangle)
    iu = np.triu_indices(n, k=1)
    pair_dists = dist_matrix[iu]

    # All distances
    all_dists = np.concatenate([pair_dists, boundary_dists])

    # Compute weights using log-sum-exp for numerical stability
    d_min = np.min(all_dists)
    shifted = all_dists - d_min
    exp_terms = np.exp(-t * shifted)
    S = np.sum(exp_terms)

    # Gradient of soft-min: f = (1/t)*log(S) + d_min
    # df/dx = -(1/S) * sum_k exp(-t*(d_k - d_min)) * d(d_k)/dx
    #        = -sum_k w_k * d(d_k)/dx  where w_k = exp(-t*(d_k-d_min))/S

    grad = np.zeros_like(pos)

    # Pairwise gradient
    pair_weights = exp_terms[:len(pair_dists)] / S
    diff_pairs = diff[iu]
    norm_pairs = pair_dists[:, np.newaxis]
    norm_pairs[norm_pairs < 1e-12] = 1e-12
    unit_vecs = diff_pairs / norm_pairs

    i_idx, j_idx = iu
    np.add.at(grad, i_idx, -pair_weights[:, np.newaxis] * unit_vecs)
    np.add.at(grad, j_idx, pair_weights[:, np.newaxis] * unit_vecs)

    # Boundary gradient
    boundary_weights = exp_terms[len(pair_dists):] / S
    norms_col = norms[:, np.newaxis]
    norms_col[norms_col < 1e-12] = 1e-12
    unit_pos = pos / norms_col
    # d(1-|x|)/dx = -x/|x|, so contribution is -w * (-unit_pos) = w * unit_pos
    grad += boundary_weights[:, np.newaxis] * unit_pos

    soft_min_val = (1.0 / t) * np.log(S) + d_min
    return grad, soft_min_val, d_min


def _local_optimize(points, max_iter=500, lr=0.005, deadline=None):
    """
    Local optimization using gradient descent on the soft-min of distances.
    Maximizes the minimum pairwise and boundary distance.
    Uses momentum and adaptive learning rate with consistent scoring.
    """
    n = len(points)
    best_points = points.copy()
    best_score = _compute_score(best_points)

    # Adaptive temperature: start moderate, increase for sharper gradient
    t_start = 30.0
    t_end = 150.0
    current_lr = lr
    momentum = np.zeros_like(points)
    beta_m = 0.85  # momentum coefficient
    patience = 0
    max_patience = 50
    score_check_interval = 3  # check exact score more frequently

    start_time = time.time()

    for iteration in range(max_iter):
        # Deadline check
        if deadline is not None and time.time() > deadline:
            break

        # Anneal temperature: increase t for sharper soft-min with smooth schedule
        progress = iteration / max(max_iter - 1, 1)
        # Use cosine annealing for smoother transition
        t = t_start + (t_end - t_start) * (1 - np.cos(np.pi * progress)) / 2

        # Compute gradient
        grad, soft_min_val, actual_min = _soft_min_gradient(best_points, t)

        # Normalize gradient to prevent large steps
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-12:
            grad = grad / grad_norm * min(grad_norm, 0.1)

        # Momentum update
        momentum = beta_m * momentum + (1 - beta_m) * grad
        new_points = best_points + current_lr * momentum

        # Keep points inside unit circle with soft constraint
        norms = np.linalg.norm(new_points, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            # Use a soft projection: pull back proportionally
            overshoot = (norms[mask] - 1.0) / norms[mask]
            new_points[mask] = new_points[mask] * (1.0 - overshoot * 0.5)
            # Hard clamp for safety
            norms_after = np.linalg.norm(new_points, axis=1)
            mask_after = norms_after > 1.0
            if np.any(mask_after):
                new_points[mask_after] = new_points[mask_after] / norms_after[mask_after, np.newaxis]

        # Compute score consistently
        score = _compute_score(new_points)

        if score > best_score:
            best_score = score
            best_points = new_points.copy()
            patience = 0
            # Slightly increase learning rate if making progress
            current_lr = min(current_lr * 1.01, 0.015)
        else:
            patience += 1
            # Gentler decay learning rate
            current_lr *= 0.99

        if patience >= max_patience:
            break

    # Final exact score
    best_score = _compute_score(best_points)
    return best_points


def _hex_lattice_init(spacing, angle_deg=0):
    """Generate hexagonal lattice points within the unit disk."""
    angle = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    k = int(1.0 / spacing) + 3
    rows = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            # Rotate
            xr = x * cos_a - y * sin_a
            yr = x * sin_a + y * cos_a
            rows.append((xr, yr))

    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    mask = radius <= 1.0 - spacing / 2.0
    points = points[mask]

    if len(points) < N:
        return None

    # Sort by distance from center and take N closest
    radius = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(radius, kind="stable")[:N]
    return points[idx]


def _radial_ring_init():
    """
    Place circles in concentric rings. Each ring has a radial distance and
    an angular offset.
    """
    best_points = None
    best_score = -np.inf

    # Try different ring structures
    for n_rings in range(5, 16):
        # Distribute N circles among rings
        # Outer rings can hold more circles
        weights = []
        for i in range(n_rings):
            r_i = (i + 1) / n_rings * 0.95
            weights.append(2 * np.pi * r_i)

        total_w = sum(weights)
        counts = []
        remaining = N
        for i in range(n_rings - 1):
            c_i = max(1, int(round(N * weights[i] / total_w)))
            c_i = min(c_i, remaining - (n_rings - 1 - i))
            counts.append(c_i)
            remaining -= c_i
        counts.append(remaining)

        # Now compute positions
        points = []
        for i, count in enumerate(counts):
            if count <= 0:
                continue
            r_i = (i + 1) / n_rings * 0.95
            # Angular offset to stagger rings
            offset = (i * np.pi) / max(1, count)
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + offset
            for a in angles:
                points.append((r_i * np.cos(a), r_i * np.sin(a)))

        if len(points) != N:
            continue

        points = np.array(points)
        score = _compute_score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()

    return best_points


def _random_init():
    """Random initialization within unit disk."""
    points = np.random.randn(N, 2)
    norms = np.linalg.norm(points, axis=1)
    points = points / norms[:, np.newaxis]
    # Scale to be within unit circle with some margin
    points *= 0.95
    return points


def _polish(points, deadline=None, max_iter=300):
    """
    High-precision polish pass with adaptive soft-min and small steps.
    Uses a multi-phase approach: first identify critical constraints, then optimize.
    """
    n = len(points)
    best_points = points.copy()
    best_score = _compute_score(best_points)

    current_lr = 0.0005
    momentum = np.zeros_like(points)
    beta_m = 0.7
    patience = 0
    max_patience = 30

    # Phase 1: Use moderate temperature to find critical pairs
    t_phase1 = 100.0
    # Phase 2: Use very sharp temperature for precise optimization
    t_phase2 = 500.0

    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        # Switch to sharper temperature in second half
        if iteration < max_iter // 2:
            t = t_phase1
        else:
            t = t_phase2

        grad, soft_min_val, actual_min = _soft_min_gradient(best_points, t)

        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-12:
            grad = grad / grad_norm * min(grad_norm, 0.05)

        momentum = beta_m * momentum + (1 - beta_m) * grad
        new_points = best_points + current_lr * momentum

        # Keep points inside unit circle
        norms = np.linalg.norm(new_points, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            new_points[mask] = new_points[mask] / norms[mask, np.newaxis]

        score = _compute_score(new_points)

        if score > best_score:
            best_score = score
            best_points = new_points.copy()
            patience = 0
            # Very gentle increase
            current_lr = min(current_lr * 1.005, 0.002)
        else:
            patience += 1
            current_lr *= 0.95

        if patience >= max_patience:
            break

    return best_points


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN

    # Generate multiple initial configurations
    configs = []

    # Hexagonal lattice with different angles - try more angles and spacings
    for angle in [0, 10, 20, 30, 40, 50, 60]:
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        for _ in range(15):
            pts = _hex_lattice_init(spacing, angle)
            if pts is not None:
                configs.append(pts)
                break
            spacing *= 0.97

    # Radial ring initialization
    ring_pts = _radial_ring_init()
    if ring_pts is not None:
        configs.append(ring_pts)

    # Random initializations
    for _ in range(2):
        configs.append(_random_init())

    # Optimize each configuration and keep the best
    best_points = None
    best_score = -np.inf
    optimized_configs = []

    for cfg in configs:
        if time.time() > deadline - 10:  # Leave time for polish
            break

        # Local optimization
        optimized = _local_optimize(cfg, max_iter=300, deadline=deadline - 10)
        score = _compute_score(optimized)

        if score > best_score:
            best_score = score
            best_points = optimized.copy()

        # Keep top 3 configurations for potential re-optimization
        if len(optimized_configs) < 3:
            optimized_configs.append((score, optimized.copy()))
        else:
            # Replace worst if this is better
            worst_idx = np.argmin([s for s, _ in optimized_configs])
            if score > optimized_configs[worst_idx][0]:
                optimized_configs[worst_idx] = (score, optimized.copy())

    # If no configuration was optimized, use the first one as fallback
    if best_points is None:
        best_points = configs[0] if configs else np.zeros((N, 2))

    # Final high-precision polish pass if time permits
    if time.time() < deadline - 3.0:
        polished = _polish(best_points, deadline=deadline, max_iter=250)
        polished_score = _compute_score(polished)
        if polished_score > best_score:
            best_points = polished
            best_score = polished_score

    # If time remains, try polishing other good configurations
    if time.time() < deadline - 2.0 and len(optimized_configs) > 1:
        for score, pts in optimized_configs[1:]:
            if time.time() > deadline - 2.0:
                break
            if score >= best_score * 0.99:  # Only polish if close to best
                polished = _polish(pts, deadline=deadline, max_iter=100)
                polished_score = _compute_score(polished)
                if polished_score > best_score:
                    best_points = polished
                    best_score = polished_score

    return best_points
# EVOLVE-BLOCK-END
