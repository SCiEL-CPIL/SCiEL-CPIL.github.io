"""Packing 300 equal circles in a unit circle.

Strategy:
1. Initialize with Fibonacci sphere distribution mapped to disk for uniform, non-lattice start.
2. Refine with enhanced local search using adaptive step sizing and bias toward close pairs.
3. Use soft penalty function with log-sum-exp for smooth surrogate optimization.
4. Final high-precision polish pass with exact radius computation.
"""
import numpy as np
import time
from scipy.spatial import cKDTree

N = 300

def _compute_radius(pts):
    """Compute the radius for a configuration of N points."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    dists_origin = np.hypot(pts[:, 0], pts[:, 1])
    boundary_min = 1.0 - np.max(dists_origin)
    if boundary_min <= 0:
        return 0.0
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    return min(min_pair / 2.0, boundary_min)

def _compute_radius_fast(pts):
    """Faster radius computation using KDTree."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    dists_origin = np.hypot(pts[:, 0], pts[:, 1])
    boundary_min = 1.0 - np.max(dists_origin)
    if boundary_min <= 0:
        return 0.0
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=2)
    min_pair = np.min(dists[:, 1])
    return min(min_pair / 2.0, boundary_min)

def _fibonacci_disk(n_points=N):
    """Generate Fibonacci sphere distribution mapped to disk."""
    # Fibonacci sphere points
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle
    y = np.linspace(1, -1, n_points)
    radius = np.sqrt(1 - y**2)
    theta = phi * np.arange(n_points)
    
    # Map to disk: use polar coordinates with uniform area distribution
    # r = sqrt(u) for uniform area, theta = golden angle * i
    u = np.linspace(0, 1, n_points)
    r = np.sqrt(u)
    theta = phi * np.arange(n_points)
    
    x = r * np.cos(theta)
    y_disk = r * np.sin(theta)
    
    return np.column_stack([x, y_disk])

def _soft_penalty_objective(pts, beta_pair=50.0, beta_boundary=50.0):
    """Compute soft penalty for overlaps and boundary violations."""
    if len(pts) != N:
        return np.inf
    
    # Boundary penalty
    dists_origin = np.hypot(pts[:, 0], pts[:, 1])
    boundary_violation = np.maximum(0, dists_origin - 1.0)
    boundary_penalty = np.sum(boundary_violation**2)
    
    # Pairwise overlap penalty using soft min
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    
    iu = np.triu_indices(N, k=1)
    dists_sq_iu = dists_sq[iu]
    
    # Soft min of squared distances
    exp_vals = np.exp(-beta_pair * dists_sq_iu)
    log_sum_exp = np.log(np.sum(exp_vals))
    soft_min_sq = -log_sum_exp / beta_pair
    soft_min_dist = np.sqrt(soft_min_sq)
    
    # Target spacing
    target_spacing = 0.1063  # ~2 * 0.05315
    overlap_violation = np.maximum(0, target_spacing - soft_min_dist)
    overlap_penalty = overlap_violation**2
    
    return boundary_penalty + overlap_penalty

def _soft_penalty_gradient(pts, beta_pair=50.0, beta_boundary=50.0):
    """Compute gradient of soft penalty objective."""
    grad = np.zeros_like(pts)
    
    # Boundary gradient
    dists_origin = np.hypot(pts[:, 0], pts[:, 1])
    boundary_violation_mask = dists_origin > 1.0
    if np.any(boundary_violation_mask):
        for i in np.where(boundary_violation_mask)[0]:
            grad[i] += 2 * (dists_origin[i] - 1.0) * pts[i] / dists_origin[i]
    
    # Pairwise gradient
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    
    iu = np.triu_indices(N, k=1)
    dists_sq_iu = dists_sq[iu]
    
    exp_vals = np.exp(-beta_pair * dists_sq_iu)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    
    # Gradient of soft min distance
    soft_min_sq = -np.log(sum_exp) / beta_pair
    soft_min_dist = np.sqrt(soft_min_sq)
    
    target_spacing = 0.1063
    if soft_min_dist < target_spacing:
        overlap_factor = 2 * (target_spacing - soft_min_dist)
        
        # Accumulate gradients for each pair
        for k in range(len(iu[0])):
            i, j = iu[0][k], iu[1][k]
            w = weights[k]
            d = diff[i, j]
            dist = np.sqrt(dists_sq_iu[k])
            if dist > 1e-10:
                grad_dir = d / dist
                grad[i] += overlap_factor * w * grad_dir
                grad[j] -= overlap_factor * w * grad_dir
    
    return grad

def _local_search(pts, deadline, seed=42, use_fast=True):
    """Enhanced local search with adaptive step sizing and bias toward close pairs."""
    rng = np.random.RandomState(seed)
    current = pts.copy()
    compute_r = _compute_radius_fast if use_fast else _compute_radius
    current_r = compute_r(current)
    best = current.copy()
    best_r = current_r
    
    step_size = 0.008
    patience = 0
    max_patience = 800
    
    while time.time() < deadline and patience < max_patience:
        tree = cKDTree(current)
        dists, _ = tree.query(current, k=2)
        close_pairs = np.argmin(dists[:, 1])
        close_dist = dists[close_pairs, 1]
        
        if close_dist < 0.11 and rng.random() < 0.7:
            idx = close_pairs
        else:
            idx = rng.randint(N)
        
        old_pos = current[idx].copy()
        best_new_r = current_r
        best_new_pos = old_pos.copy()
        
        n_attempts = 8 if patience < 100 else 5
        for attempt in range(n_attempts):
            effective_step = step_size * (1.0 + 0.5 * (1.0 - current_r / 0.055))
            new_pos = old_pos + rng.uniform(-effective_step, effective_step, 2)
            
            new_r_mag = np.hypot(new_pos[0], new_pos[1])
            if new_r_mag > 0.995:
                new_pos = new_pos / new_r_mag * 0.995
            
            current[idx] = new_pos
            new_r = compute_r(current)
            
            if new_r > best_new_r:
                best_new_r = new_r
                best_new_pos = new_pos.copy()
        
        current[idx] = best_new_pos
        
        if best_new_r > current_r + 1e-12:
            current_r = best_new_r
            patience = 0
            if current_r > best_r:
                best_r = current_r
                best = current.copy()
        else:
            current[idx] = old_pos
            patience += 1
            if patience % 30 == 0:
                step_size *= 0.93
            if step_size < 1e-6:
                break
    
    return best, best_r

def _soft_penalty_relax(pts, deadline, seed=123):
    """Relaxation using soft penalty function with gradient descent."""
    rng = np.random.RandomState(seed)
    current = pts.copy()
    current_r = _compute_radius_fast(current)
    best = current.copy()
    best_r = current_r
    
    lr = 0.001
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        obj = _soft_penalty_objective(current)
        grad = _soft_penalty_gradient(current)
        
        # Gradient descent step
        current = current - lr * grad
        
        # Keep inside circle
        r_mags = np.hypot(current[:, 0], current[:, 1])
        mask = r_mags > 0.998
        if np.any(mask):
            current[mask] = current[mask] / r_mags[mask, np.newaxis] * 0.998
        
        new_r = _compute_radius_fast(current)
        if new_r > current_r + 1e-12:
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best = current.copy()
        
        # Adaptive learning rate
        if iteration % 50 == 0:
            lr *= 0.9
    
    return best, best_r

def _repulsion_relax(pts, deadline, seed=123):
    """Repulsion-based relaxation to untangle close pairs."""
    rng = np.random.RandomState(seed)
    current = pts.copy()
    current_r = _compute_radius_fast(current)
    best = current.copy()
    best_r = current_r
    
    for iteration in range(1000):
        if time.time() > deadline:
            break
        
        tree = cKDTree(current)
        dists, pairs = tree.query(current, k=2)
        
        min_dist = np.min(dists[:, 1])
        target_spacing = 2 * (0.0532)
        if min_dist > target_spacing:
            break
        
        close_mask = dists[:, 1] < target_spacing
        close_indices = np.where(close_mask)[0]
        
        if len(close_indices) == 0:
            break
        
        n_to_process = min(5, len(close_indices))
        sorted_idx = close_indices[np.argsort(dists[close_indices, 1])][:n_to_process]
        
        for min_idx in sorted_idx:
            i, j = pairs[min_idx, 0], pairs[min_idx, 1]
            if i == j:
                continue
            
            direction = current[i] - current[j]
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                direction = rng.randn(2)
                norm = np.linalg.norm(direction)
            direction = direction / norm
            
            close_ratio = dists[min_idx, 1] / target_spacing
            push = 0.004 * max(0.1, 1.0 - close_ratio)
            current[i] += direction * push
            current[j] -= direction * push
            
            for k in [i, j]:
                r_mag = np.hypot(current[k, 0], current[k, 1])
                if r_mag > 0.998:
                    current[k] = current[k] / r_mag * 0.998
        
        new_r = _compute_radius_fast(current)
        if new_r > current_r + 1e-12:
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best = current.copy()
    
    return best, best_r

def construct():
    deadline = time.time() + 140.0
    
    def time_check():
        return time.time() < deadline
    
    # Stage 1: Initialize with Fibonacci disk distribution
    best_points = _fibonacci_disk(N)
    best_radius = _compute_radius(best_points)
    
    # Also try hexagonal lattice for comparison
    sqrt3 = np.sqrt(3.0)
    for trial in range(20):
        if not time_check():
            break
        spacing = 0.104 + trial * 0.001
        k = int(1.0 / spacing) + 3
        points_list = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * sqrt3 / 2.0
                r = np.hypot(x, y)
                if r < 1.0:
                    points_list.append((x, y))
        
        if len(points_list) >= N:
            points = np.array(points_list)
            dists = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(dists, kind='stable')[:N]
            pts = points[idx]
            r = _compute_radius(pts)
            if r > best_radius:
                best_radius = r
                best_points = pts.copy()
    
    # Stage 2: Soft penalty relaxation
    soft_budget = deadline - 60.0
    soft_pts, soft_r = _soft_penalty_relax(best_points, soft_budget, seed=123)
    if soft_r > best_radius:
        best_points = soft_pts
        best_radius = soft_r
    
    # Stage 3: Repulsion relaxation
    rep_budget = deadline - 30.0
    rep_pts, rep_r = _repulsion_relax(best_points, rep_budget, seed=123)
    if rep_r > best_radius:
        best_points = rep_pts
        best_radius = rep_r
    
    # Stage 4: Local search with multiple seeds
    for seed in [42, 777, 999, 1234]:
        if not time_check():
            break
        ls_budget = deadline - 15.0
        ls_pts, ls_r = _local_search(best_points, ls_budget, seed=seed, use_fast=True)
        if ls_r > best_radius:
            best_points = ls_pts
            best_radius = ls_r
    
    # Stage 5: Final high-precision polish
    if time_check():
        polish_budget = deadline - 3.0
        polish_pts, polish_r = _local_search(best_points, polish_budget, seed=12345, use_fast=False)
        if polish_r > best_radius:
            best_points = polish_pts
            best_radius = polish_r
    
    # Final validation
    dists_origin = np.hypot(best_points[:, 0], best_points[:, 1])
    max_allowed = 1.0 - 1e-10
    mask = dists_origin >= max_allowed
    if np.any(mask):
        best_points[mask] *= (max_allowed / dists_origin[mask]) * 0.999
    
    tree = cKDTree(best_points)
    dists, pairs = tree.query(best_points, k=2)
    close_mask = dists[:, 1] < 1e-6
    if np.any(close_mask):
        close_indices = np.where(close_mask)[0]
        rng_fix = np.random.RandomState(999)
        for idx in close_indices:
            j = pairs[idx, 1]
            angle = rng_fix.uniform(0, 2 * np.pi)
            best_points[idx, 0] += 0.01 * np.cos(angle)
            best_points[idx, 1] += 0.01 * np.sin(angle)
            best_points[j, 0] -= 0.01 * np.cos(angle)
            best_points[j, 1] -= 0.01 * np.sin(angle)
    
    if not np.all(np.isfinite(best_points)) or best_points.shape != (N, 2):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = np.sqrt(np.linspace(0, 1, N))
        best_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        best_points *= 0.99
    
    return best_points
