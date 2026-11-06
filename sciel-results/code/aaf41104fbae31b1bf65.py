import numpy as np
import time
from scipy.spatial.distance import cdist

N = 200
DEADLINE = 145.0  # Safety margin before 150s limit


def _compute_radius(points):
    """Compute the maximum common radius for given circle centers."""
    if points.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    if np.any(points[:, 0] < -0.5) or np.any(points[:, 0] > 0.5):
        return 0.0
    if np.any(points[:, 1] < -0.5) or np.any(points[:, 1] > 0.5):
        return 0.0

    # Center-center distances
    dists = cdist(points, points)
    np.fill_diagonal(dists, np.inf)
    min_center_dist = np.min(dists)

    # Wall distances
    dx = 0.5 - np.abs(points[:, 0])
    dy = 0.5 - np.abs(points[:, 1])
    min_wall_dist = np.min(np.concatenate([dx, dy]))

    r = min(min_center_dist, min_wall_dist) / 2.0
    return r


def _generate_hex_grid(n):
    """Generate a hexagonal grid of points inside the unit square."""
    cols = 15
    rows = 14
    s = 1.0 / (cols + 0.5)
    
    points = []
    for r in range(rows):
        for c in range(cols):
            x = -0.5 + (c + 0.5) * s
            y = -0.5 + (r + 0.5) * s * np.sqrt(3) / 2
            if r % 2 != 0:
                x += s / 2.0
            points.append([x, y])
    
    points = np.array(points)
    
    mask = (points[:, 0] >= -0.5) & (points[:, 0] <= 0.5) & \
           (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
    points = points[mask]
    
    if len(points) > n:
        # Select n points greedily by max min-distance
        selected = [0]
        remaining = list(range(1, len(points)))
        while len(selected) < n and remaining:
            best_idx = -1
            best_min_dist = -1
            for idx in remaining:
                min_d = min(np.linalg.norm(points[idx] - points[s]) for s in selected)
                if min_d > best_min_dist:
                    best_min_dist = min_d
                    best_idx = idx
            if best_idx >= 0:
                selected.append(best_idx)
                remaining.remove(best_idx)
            else:
                break
        points = points[selected]
    elif len(points) < n:
        missing = n - len(points)
        random_pts = np.random.uniform(-0.45, 0.45, (missing, 2))
        points = np.vstack([points, random_pts])
    
    return points


def _objective_and_grad(x, r_target):
    """
    Objective: soft-min of all pair distances and wall distances.
    We want to maximize the minimum distance, so we minimize the negative soft-min.
    Using a smooth approximation: -softmin(d_i) where softmin is a smooth minimum.
    
    For 200 points, we have ~20000 pairs. We use a log-sum-exp trick for softmin.
    """
    points = x.reshape(-1, 2)
    
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    
    # Wall distances
    dx = 0.5 - np.abs(points[:, 0])
    dy = 0.5 - np.abs(points[:, 1])
    wall_dists = np.concatenate([dx, dy])
    
    # All distances
    all_dists = np.concatenate([dist[np.triu_indices(N, k=1)], wall_dists])
    
    # Soft-min: use log-sum-exp with negative sign
    # softmin(a) = -logsumexp(-a / t) * t for temperature t
    # We want to maximize softmin, so minimize -softmin = logsumexp(-a/t)*t
    t = 0.005  # temperature for soft-min
    neg_dists = -all_dists / t
    # logsumexp for numerical stability
    max_val = np.max(neg_dists)
    logsumexp_val = max_val + np.log(np.sum(np.exp(neg_dists - max_val)))
    softmin_val = -logsumexp_val * t
    
    # Objective: we want to maximize softmin, so return -softmin
    obj = -softmin_val
    
    # Gradient computation
    # d(softmin)/d(all_dists[i]) = exp(-all_dists[i]/t) / sum(exp(-all_dists[j]/t))
    # Then chain rule back to points
    
    weights = np.exp(neg_dists - max_val)
    weights /= np.sum(weights)
    
    # Gradient w.r.t. all_dists
    grad_all_dists = -weights / t  # d(-softmin)/d(all_dists)
    
    # Gradient w.r.t. pair distances
    n_pairs = N * (N - 1) // 2
    grad_dist = grad_all_dists[:n_pairs]
    grad_wall = grad_all_dists[n_pairs:]
    
    # Gradient w.r.t. points from pair distances
    # d(dist_ij)/d(points[i]) = (points[i] - points[j]) / dist_ij
    grad_points = np.zeros_like(points)
    
    # This is expensive to do fully, so we approximate
    # For each pair (i,j), contribution to grad_points[i] is grad_dist[k] * (points[i]-points[j])/dist_ij
    # and to grad_points[j] is -grad_dist[k] * (points[i]-points[j])/dist_ij
    
    # We'll do this in a loop for clarity, but it's O(N^2)
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            d = dist[i, j]
            if d < 1e-10:
                continue
            w = grad_dist[idx]
            direction = (points[i] - points[j]) / d
            grad_points[i] += w * direction
            grad_points[j] -= w * direction
            idx += 1
    
    # Gradient w.r.t. wall distances
    # dx_i = 0.5 - |x_i|, so d(dx_i)/d(x_i) = -sign(x_i)
    # dy_i = 0.5 - |y_i|, so d(dy_i)/d(y_i) = -sign(y_i)
    for i in range(N):
        grad_points[i, 0] += grad_wall[i] * (-np.sign(points[i, 0]))
        grad_points[i, 1] += grad_wall[N + i] * (-np.sign(points[i, 1]))
    
    return obj, grad_points.flatten()


def _optimize_smooth(points, r_target, n_iters, start_time, deadline, 
                     lr=0.001, decay=0.9995):
    """
    Gradient descent using smooth objective.
    """
    points = points.copy().astype(np.float64)
    x = points.flatten()
    lr_current = lr
    
    for it in range(n_iters):
        if time.time() > deadline - 5:
            break
        
        obj, grad = _objective_and_grad(x, r_target)
        
        # Gradient descent step
        x -= lr_current * grad
        
        # Reshape and clip
        points = x.reshape(-1, 2)
        points[:, 0] = np.clip(points[:, 0], -0.5 + 1e-6, 0.5 - 1e-6)
        points[:, 1] = np.clip(points[:, 1], -0.5 + 1e-6, 0.5 - 1e-6)
        x = points.flatten()
        
        lr_current *= decay
    
    return points


def construct():
    start_time = time.time()
    np.random.seed(42)
    
    # Generate initial hex grid
    init_points = _generate_hex_grid(N)
    init_points = np.clip(init_points, -0.48, 0.48)
    
    best_points = init_points.copy()
    best_score = _compute_radius(init_points)
    
    # Multi-scale optimization
    r_start = 0.02
    
    r_targets = []
    r = r_start
    while r < 0.0365:
        r_targets.append(r)
        r *= 1.04
    r_targets.append(0.0365)
    r_targets.append(0.0366)
    r_targets.append(0.0367)
    
    for r_target in r_targets:
        if time.time() > DEADLINE - 40:
            break
        
        # Run smooth optimization
        points = _optimize_smooth(best_points, r_target,
                                   n_iters=1000,
                                   start_time=start_time,
                                   deadline=DEADLINE,
                                   lr=0.002,
                                   decay=0.9997)
        
        actual_r = _compute_radius(points)
        
        if actual_r > best_score:
            best_score = actual_r
            best_points = points.copy()
    
    # Final polish
    if time.time() < DEADLINE - 10:
        remaining_time = DEADLINE - 10 - time.time()
        n_final_iters = int(min(2000, max(200, remaining_time * 20)))
        final_points = _optimize_smooth(best_points, best_score,
                                         n_iters=n_final_iters,
                                         start_time=start_time,
                                         deadline=DEADLINE,
                                         lr=0.0003,
                                         decay=0.9999)
        final_score = _compute_radius(final_points)
        if final_score > best_score:
            best_score = final_score
            best_points = final_points
    
    # Ensure all points are strictly inside
    best_points[:, 0] = np.clip(best_points[:, 0], -0.49999, 0.49999)
    best_points[:, 1] = np.clip(best_points[:, 1], -0.49999, 0.49999)
    
    return best_points
