"""Fresh design: soft-min surrogate + gradient ascent for 300-circle packing."""
import numpy as np
import time

N = 300


def _softmin(vals, beta):
    """Soft-min of an array via log-sum-exp: -logsumexp(-beta*x)/beta."""
    m = np.min(vals)
    # Clip to avoid overflow
    shifted = vals - m
    logsum = np.log(np.sum(np.exp(-beta * shifted))) + (-beta * m)
    return -logsum / beta


def _compute_softmin_obj(pts, beta):
    """
    Compute soft-min of all pair distances and (1 - |c_i|) distances.
    Returns (softmin_value, grad).
    """
    n = len(pts)
    # Pair distances
    # Use chunked computation to save memory
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=-1))
    # Set diagonal to large value so it doesn't affect min
    np.fill_diagonal(dists, 1e10)
    # Boundary distances
    radii = np.sqrt(np.sum(pts * pts, axis=1))
    bdist = 1.0 - radii

    # Combine: we want soft-min over all pair dists and all boundary dists
    # For efficiency, compute soft-min separately and combine
    # Actually, let's compute the combined soft-min properly
    # All values: pair dists (lower triangle) and boundary dists
    # For gradient, we need d(softmin)/d(pts)
    
    # Let's use a combined approach:
    # f = softmin(all_vals) where all_vals includes pair dists and boundary dists
    # For memory, compute in chunks
    
    # Pair distance gradient contribution
    # For softmin of dists, the weight for each pair is exp(-beta * (d - m)) / sum(exp(-beta*(d-m)))
    # where m = min of all values
    
    # Get all values for finding min
    lower_idx = np.tril_indices(n, k=-1)
    pair_dists = dists[lower_idx]
    all_vals = np.concatenate([pair_dists, bdist])
    m = np.min(all_vals)
    
    # Weights for soft-min
    # exp(-beta * (val - m))
    pair_weights = np.exp(-beta * (pair_dists - m))
    b_weights = np.exp(-beta * (bdist - m))
    
    total = np.sum(pair_weights) + np.sum(b_weights)
    
    # softmin value
    softmin_val = m + np.log(total) / beta
    
    # Gradient
    grad = np.zeros_like(pts)
    
    # Pair contribution
    # d(pair_dist)/d(pts_i) = (pts_i - pts_j) / pair_dist
    # Contribution to grad[i] from pair (i,j): -weight_ij * (pts_i - pts_j) / d_ij / total * beta
    # Wait, let me re-derive.
    # softmin = -log(sum(exp(-beta * vals))) / beta
    # d(softmin)/d(val_k) = (beta * exp(-beta*val_k)) / (beta * sum(exp(-beta*vals)))
    # = exp(-beta*val_k) / sum(exp(-beta*vals))
    # So weight_k = exp(-beta * val_k) / total
    # And d(softmin)/d(pts) = sum_k weight_k * d(val_k)/d(pts)
    
    # For pair (i,j), val = dist(i,j), d(dist)/d(pts_i) = (pts_i - pts_j)/dist
    # weight for pair (i,j) = pair_weights[idx] / total
    
    # Vectorized: for each pair, add to grad
    # This is O(n^2) which for n=300 is 45000 pairs, manageable
    
    # Build the pair index arrays
    i_idx, j_idx = lower_idx
    d_vec = pts[i_idx] - pts[j_idx]  # shape (num_pairs, 2)
    d_norm = pair_dists  # shape (num_pairs,)
    
    # weights normalized
    w_pair = pair_weights / total  # shape (num_pairs,)
    
    # grad[i] += w_pair * d_vec / d_norm
    # grad[j] -= w_pair * d_vec / d_norm
    np.add.at(grad, i_idx, w_pair[:, np.newaxis] * d_vec / d_norm[:, np.newaxis])
    np.add.at(grad, j_idx, -w_pair[:, np.newaxis] * d_vec / d_norm[:, np.newaxis])
    
    # Boundary contribution
    # val = 1 - |pts_i|, d(val)/d(pts_i) = -pts_i / |pts_i|
    w_b = b_weights / total
    radii_safe = np.where(radii > 1e-12, radii, 1e-12)
    grad += w_b[:, np.newaxis] * (-pts / radii_safe[:, np.newaxis])
    
    return softmin_val, grad


def _compute_true_score(pts):
    """Compute the actual (nonsmooth) score."""
    n = len(pts)
    if n < 2:
        return 0.0
    # Pair distances
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Boundary
    radii = np.sqrt(np.sum(pts * pts, axis=1))
    min_boundary = np.min(1.0 - radii)
    
    score = min(min_pair, min_boundary)
    if not np.isfinite(score) or score < 0:
        return 0.0
    return score


def _hex_init(n, rotation=0.0):
    """Generate a hexagonal lattice initial placement."""
    # Estimate spacing
    # Area per circle ~ pi*r^2, total area ~ pi, so r ~ 1/sqrt(n)
    # Spacing ~ 2r
    r_est = 1.0 / np.sqrt(n)
    spacing = 2.0 * r_est * 1.05  # slightly larger to ensure fit
    
    points = []
    k = int(1.0 / spacing) + 3
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            points.append([x, y])
    
    points = np.array(points)
    radii = np.hypot(points[:, 0], points[:, 1])
    # Keep points well inside
    margin = spacing * 0.5
    mask = radii <= 1.0 - margin
    points = points[mask]
    
    # If we have more than n, select n closest to center
    if len(points) >= n:
        radii = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(radii, kind='stable')[:n]
        points = points[idx]
    elif len(points) < n:
        # Add more points by reducing spacing
        spacing *= 0.95
        return _hex_init(n, rotation)
    
    # Apply rotation
    if rotation != 0.0:
        c, s = np.cos(rotation), np.sin(rotation)
        rot = np.array([[c, -s], [s, c]])
        points = points @ rot.T
    
    return points


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s margin before 150s kill
    
    # Initial placement
    pts = _hex_init(N, rotation=np.pi / 12)
    
    # Ensure valid shape
    if len(pts) != N:
        # Pad or trim
        if len(pts) < N:
            # Add random points near center
            extra = np.random.randn(N - len(pts), 2) * 0.1
            pts = np.vstack([pts, extra])
        else:
            pts = pts[:N]
    
    best_score = _compute_true_score(pts)
    best_pts = pts.copy()
    
    # Multi-stage soft-min gradient ascent
    # Beta schedule: start large (smooth), end small (sharp)
    # Actually, for soft-min, large beta -> sharp min, small beta -> smooth
    # We want to start smooth (small beta) and end sharp (large beta)
    
    stages = [
        # (beta, steps, lr)
        (10.0, 50, 0.001),
        (50.0, 50, 0.0005),
        (200.0, 50, 0.0002),
        (1000.0, 50, 0.00005),
        (5000.0, 50, 0.00001),
        (20000.0, 30, 0.000002),
    ]
    
    for beta, steps, lr in stages:
        if time.time() > deadline - 10.0:
            break
        
        for step in range(steps):
            if time.time() > deadline - 5.0:
                break
            
            val, grad = _compute_softmin_obj(pts, beta)
            
            # Gradient ascent (maximize softmin)
            pts = pts + lr * grad
            
            # Clip to unit circle with small margin
            radii = np.sqrt(np.sum(pts * pts, axis=1))
            mask = radii > 0.999
            if np.any(mask):
                pts[mask] *= 0.999 / radii[mask, np.newaxis]
        
        # Evaluate true score after each stage
        score = _compute_true_score(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    # Final polish: a few steps with very high beta
    if time.time() < deadline - 2.0:
        beta = 50000.0
        for step in range(20):
            if time.time() > deadline - 1.0:
                break
            val, grad = _compute_softmin_obj(best_pts, beta)
            new_pts = best_pts + 1e-6 * grad
            radii = np.sqrt(np.sum(new_pts * new_pts, axis=1))
            mask = radii > 0.9999
            if np.any(mask):
                new_pts[mask] *= 0.9999 / radii[mask, np.newaxis]
            s = _compute_true_score(new_pts)
            if s > best_score:
                best_score = s
                best_pts = new_pts
    
    # Return best
    return best_pts
