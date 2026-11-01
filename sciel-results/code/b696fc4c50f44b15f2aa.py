"""Improved packing of 300 equal circles in a unit circle using soft-min gradient descent."""
import numpy as np
import time

N = 300
DEADLINE = 115.0  # safety margin before 120s kill


def _compute_score(pts):
    """Compute the score for a set of points."""
    # Distance to boundary
    radii = np.linalg.norm(pts, axis=1)
    boundary_scores = 1.0 - radii
    
    # Pairwise distances
    # For 300 points, we have 300*299/2 = 44850 pairs
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Set diagonal to inf
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists, axis=1)
    pair_scores = min_pair_dist / 2.0
    
    # The radius is the minimum of all boundary scores and pair scores
    all_scores = np.concatenate([boundary_scores, pair_scores])
    return np.min(all_scores)


def _soft_min(x, beta):
    """Compute soft minimum using log-sum-exp trick."""
    # soft_min(x) ≈ -(1/beta) * log(sum(exp(-beta*x_i)))
    # For numerical stability, subtract max
    x_neg = -beta * x
    x_max = np.max(x_neg)
    # If x_max is very large positive, exp(x_neg - x_max) could overflow for the max term
    # But we're subtracting x_max, so the max term is exp(0)=1
    # Other terms are exp(x_neg - x_max) <= 1, so no overflow
    # However if x_max is very negative, exp(x_neg - x_max) could underflow, which is fine
    exp_sum = np.sum(np.exp(x_neg - x_max))
    return -(x_max + np.log(exp_sum)) / beta


def _soft_min_grad(x, beta):
    """Compute gradient of soft minimum."""
    x_neg = -beta * x
    x_max = np.max(x_neg)
    exp_vals = np.exp(x_neg - x_max)
    exp_sum = np.sum(exp_vals)
    weights = exp_vals / exp_sum
    # d/dx_i soft_min = -beta * (-weights_i) / beta = weights_i ... wait
    # soft_min = -(1/beta) * log(sum(exp(-beta*x)))
    # d/dx_i = -(1/beta) * (1/sum(exp(-beta*x))) * (-beta * exp(-beta*x_i))
    # = exp(-beta*x_i) / sum(exp(-beta*x_j))
    # = weights_i
    return weights


def construct():
    start_time = time.time()
    
    # Initialize with a hexagonal lattice pattern
    np.random.seed(42)
    
    # Create initial hexagonal lattice
    spacing = 0.055
    points = []
    k = int(1.0 / spacing) + 2
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 1.0:
                points.append([x, y])
    
    points = np.array(points)
    
    # If we have more than N, randomly sample N
    if len(points) > N:
        idx = np.random.choice(len(points), N, replace=False)
        points = points[idx]
    
    # If we have fewer, add random points
    while len(points) < N:
        angle = np.random.uniform(0, 2*np.pi)
        r = np.random.uniform(0, 0.95)
        points = np.vstack([points, [r*np.cos(angle), r*np.sin(angle)]])
    
    points = points[:N]
    
    best_score = _compute_score(points)
    best_points = points.copy()
    
    # Gradient descent with soft-min objective
    # We want to maximize the minimum distance, so minimize negative of soft-min
    
    beta_start = 100.0
    beta_end = 10000.0
    n_steps = 5000
    lr_start = 0.001
    lr_end = 0.0001
    
    for step in range(n_steps):
        if time.time() - start_time > DEADLINE:
            break
        
        # Schedule beta
        t = step / n_steps
        beta = beta_start * (beta_end / beta_start) ** t
        lr = lr_start * (lr_end / lr_start) ** t
        
        # Compute soft-min of all constraints
        radii = np.linalg.norm(points, axis=1)
        boundary_scores = 1.0 - radii
        
        # Pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        pair_scores = dists / 2.0
        
        # Flatten pair scores (upper triangle only to avoid double counting)
        i_idx, j_idx = np.triu_indices(N, k=1)
        pair_scores_flat = pair_scores[i_idx, j_idx]
        
        all_scores = np.concatenate([boundary_scores, pair_scores_flat])
        
        # Soft min and its gradient
        soft_min_val = _soft_min(all_scores, beta)
        weights = _soft_min_grad(all_scores, beta)
        
        # Gradient with respect to points
        grad = np.zeros_like(points)
        
        # Boundary constraint gradients
        # d/dx_i (1 - |p_i|) = -p_i / |p_i|
        for i in range(N):
            r = radii[i]
            if r > 1e-10:
                grad[i] -= weights[i] * points[i] / r
        
        # Pair constraint gradients
        # d/dx_i (|p_i - p_j|/2) = (p_i - p_j) / (2 * |p_i - p_j|)
        # d/dx_j (|p_i - p_j|/2) = (p_j - p_i) / (2 * |p_i - p_j|)
        for k in range(len(i_idx)):
            i, j = i_idx[k], j_idx[k]
            d = dists[i, j]
            if d > 1e-10:
                w = weights[N + k]
                grad[i] += w * (points[i] - points[j]) / d
                grad[j] += w * (points[j] - points[i]) / d
        
        # Update
        points -= lr * grad
        
        # Project back: ensure points are inside the unit circle
        radii = np.linalg.norm(points, axis=1)
        mask = radii > 1.0
        if np.any(mask):
            points[mask] = points[mask] / radii[mask, np.newaxis] * (1.0 - 1e-6)
        
        # Track best
        current_score = _compute_score(points)
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
    
    return best_points


# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
