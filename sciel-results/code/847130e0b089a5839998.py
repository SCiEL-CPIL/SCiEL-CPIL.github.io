"""Packing 300 equal circles in a unit circle.

Strategy:
1. Generate hexagonal lattice initial configurations with multiple spacings.
2. Refine with fast local search using KDTree-based radius evaluation.
3. Multi-pair repulsion relaxation to untangle close pairs simultaneously.
4. Deadline guard ensures we return the best configuration found before
   the 150 s wall-clock kill.
"""
import numpy as np
import time
from scipy.spatial import cKDTree

N = 300

def _compute_radius(pts):
    """Compute the radius for a configuration of N points (exact, O(N^2))."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Boundary constraint
    dists_origin = np.hypot(pts[:, 0], pts[:, 1])
    boundary_min = 1.0 - np.max(dists_origin)
    if boundary_min <= 0:
        return 0.0
    # Pairwise distances using vectorized approach
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    return min(min_pair / 2.0, boundary_min)


def _compute_radius_fast(pts):
    """Faster radius computation using KDTree nearest-neighbor queries."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    dists_origin = np.hypot(pts[:, 0], pts[:, 1])
    boundary_min = 1.0 - np.max(dists_origin)
    if boundary_min <= 0:
        return 0.0
    # Use scipy's cKDTree for efficient nearest neighbor
    tree = cKDTree(pts)
    # Get 2 nearest neighbors for each point (one is self, so we need 2)
    dists, _ = tree.query(pts, k=2)
    min_pair = np.min(dists[:, 1])  # exclude self-distance
    return min(min_pair / 2.0, boundary_min)


def _hex_initial(spacing, n_points=N, rotation=0.0):
    """Generate hexagonal lattice points with optional rotation, select best n_points."""
    sqrt3 = np.sqrt(3.0)
    k = int(1.0 / spacing) + 3
    
    points_list = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * sqrt3 / 2.0
            # Apply rotation
            if rotation != 0.0:
                cos_r, sin_r = np.cos(rotation), np.sin(rotation)
                x_new = x * cos_r - y * sin_r
                y_new = x * sin_r + y * cos_r
                x, y = x_new, y_new
            r = np.hypot(x, y)
            if r < 1.0:
                points_list.append((x, y))
    
    if len(points_list) < n_points:
        return None, 0.0
    
    points = np.array(points_list)
    dists = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(dists, kind='stable')[:n_points]
    pts = points[idx]
    r = _compute_radius_fast(pts)
    return pts, r


def _local_search(pts, deadline, seed=42, use_fast=True):
    """Local search: perturb random points, accept if radius improves."""
    rng = np.random.RandomState(seed)
    current = pts.copy()
    compute_r = _compute_radius_fast if use_fast else _compute_radius
    current_r = compute_r(current)
    best = current.copy()
    best_r = current_r
    
    step_size = 0.008
    patience = 0
    max_patience = 3000
    n_attempts = 10
    
    while time.time() < deadline and patience < max_patience:
        idx = rng.randint(N)
        old_pos = current[idx].copy()
        
        # Try multiple perturbations and pick the best
        best_new_r = current_r
        best_new_pos = old_pos.copy()
        
        for attempt in range(n_attempts):
            # Adaptive perturbation: larger when stuck
            scale = step_size * (1.0 + 0.5 * (patience / 100.0))
            new_pos = old_pos + rng.uniform(-scale, scale, 2)
            
            # Keep inside circle
            new_r_mag = np.hypot(new_pos[0], new_pos[1])
            if new_r_mag > 0.999:
                new_pos = new_pos / new_r_mag * 0.999
            
            current[idx] = new_pos
            new_r = compute_r(current)
            
            if new_r > best_new_r:
                best_new_r = new_r
                best_new_pos = new_pos.copy()
        
        current[idx] = best_new_pos
        
        if best_new_r > current_r + 1e-13:
            current_r = best_new_r
            patience = 0
            if current_r > best_r:
                best_r = current_r
                best = current.copy()
        else:
            current[idx] = old_pos
            patience += 1
            # Adaptive step size
            if patience % 100 == 0:
                step_size *= 0.92
            if step_size < 1e-6:
                break
    
    return best, best_r


def _repulsion_relax(pts, deadline, seed=123):
    """Repulsion-based relaxation: push apart multiple close pairs simultaneously."""
    rng = np.random.RandomState(seed)
    current = pts.copy()
    current_r = _compute_radius_fast(current)
    best = current.copy()
    best_r = current_r
    
    target_spacing = 0.106  # slightly below expected optimal spacing
    
    for iteration in range(2000):
        if time.time() > deadline:
            break
        
        # Find all close pairs using KDTree
        tree = cKDTree(current)
        dists, pairs = tree.query(current, k=2)
        
        # Identify pairs that are too close
        close_threshold = current_r * 2.0 + 0.001
        close_mask = dists[:, 1] < close_threshold
        n_close = np.sum(close_mask)
        
        if n_close == 0:
            break
        
        # Apply repulsion to all close pairs
        forces = np.zeros_like(current)
        for idx in np.where(close_mask)[0]:
            i, j = pairs[idx, 0], pairs[idx, 1]
            direction = current[i] - current[j]
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                direction = rng.randn(2)
                norm = np.linalg.norm(direction)
            direction = direction / norm
            # Force proportional to how close they are
            strength = max(0.002, (close_threshold - dists[idx, 1]) * 0.1)
            forces[i] += direction * strength
            forces[j] -= direction * strength
        
        # Apply forces
        current += forces
        
        # Keep inside circle
        r_mags = np.hypot(current[:, 0], current[:, 1])
        outside = r_mags > 0.999
        if np.any(outside):
            current[outside] = current[outside] / r_mags[outside, np.newaxis] * 0.999
        
        new_r = _compute_radius_fast(current)
        if new_r > current_r + 1e-13:
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best = current.copy()
    
    return best, best_r


def _gradient_ascent(pts, deadline, seed=456):
    """Gradient ascent on a smooth surrogate using KDTree for efficiency."""
    rng = np.random.RandomState(seed)
    current = pts.copy()
    current_r = _compute_radius_fast(current)
    best = current.copy()
    best_r = current_r
    
    beta = 30.0  # softness parameter
    
    def compute_soft_objective_and_grad(x):
        """Compute soft-min objective and gradient using KDTree for speed."""
        x = x.reshape(-1, 2)
        
        # Boundary term
        dists_origin = np.hypot(x[:, 0], x[:, 1])
        boundary_term = np.max(dists_origin)
        
        # Use KDTree to find nearest neighbors and compute soft min
        tree = cKDTree(x)
        # Query for k=6 nearest neighbors to get good approximation
        k = min(6, N)
        dists, indices = tree.query(x, k=k)
        
        # Exclude self-distance (first column)
        nn_dists = dists[:, 1:]  # shape (N, k-1)
        
        # Soft min of nearest neighbor distances
        # For each point, compute soft min over its k-1 nearest neighbors
        beta_nn = beta * 2.0
        # Weight by exp(-beta * dist^2)
        weights = np.exp(-beta_nn * nn_dists ** 2)
        weights_sum = np.sum(weights, axis=1, keepdims=True) + 1e-30
        weights /= weights_sum
        
        # Soft min distance per point
        soft_min_per_point = np.sum(weights * nn_dists, axis=1)
        
        # Overall soft min is the min over all points
        # Use soft min over the per-point values
        beta_global = 20.0
        log_weights = -beta_global * soft_min_per_point
        log_weights -= np.max(log_weights)  # for numerical stability
        global_weights = np.exp(log_weights)
        global_weights_sum = np.sum(global_weights) + 1e-30
        global_weights /= global_weights_sum
        
        soft_min_dist = np.sum(global_weights * soft_min_per_point)
        
        # Objective: -min(soft_min_dist/2, 1 - boundary_term)
        pair_term = soft_min_dist / 2.0
        boundary_term_val = 1.0 - boundary_term
        neg_radius = -min(pair_term, boundary_term_val)
        
        # Gradient
        grad = np.zeros_like(x)
        
        if pair_term < boundary_term_val:
            # Gradient from pairwise term
            # d(soft_min)/d(x_i) = sum_j w_ij * (x_i - x_j) / dist_ij * ...
            # Simplified: for each point, gradient is weighted sum of vectors to neighbors
            for i in range(N):
                gw = global_weights[i]
                if gw < 1e-10:
                    continue
                for j_idx in range(k - 1):
                    j = indices[i, j_idx + 1]
                    w = weights[i, j_idx] * gw
                    if w < 1e-10:
                        continue
                    diff = x[i] - x[j]
                    dist = nn_dists[i, j_idx] + 1e-10
                    grad[i] += w * diff / dist
                    grad[j] -= w * diff / dist
            grad /= 2.0
        else:
            # Gradient from boundary term
            max_idx = np.argmax(dists_origin)
            if dists_origin[max_idx] > 1e-10:
                grad[max_idx] = x[max_idx] / dists_origin[max_idx]
        
        return neg_radius, grad
    
    # Simple gradient ascent with line search
    lr = 0.01
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        obj, grad = compute_soft_objective_and_grad(current)
        
        # Gradient ascent step
        new_x = current + lr * grad
        
        # Keep inside circle
        r_mags = np.hypot(new_x[:, 0], new_x[:, 1])
        outside = r_mags > 0.999
        if np.any(outside):
            new_x[outside] = new_x[outside] / r_mags[outside, np.newaxis] * 0.999
        
        new_r = _compute_radius_fast(new_x)
        if new_r > current_r + 1e-13:
            current = new_x
            current_r = new_r
            lr = min(lr * 1.1, 0.05)
            if current_r > best_r:
                best_r = current_r
                best = current.copy()
        else:
            lr *= 0.8
            if lr < 1e-5:
                break
    
    return best, best_r


def construct():
    deadline = time.time() + 145.0  # safety margin before 150s kill
    
    def time_check():
        return time.time() < deadline
    
    # --- Stage 1: Construct a good initial configuration ---
    best_points = None
    best_radius = 0.0
    
    # Try multiple spacings and rotations
    spacings = np.arange(0.075, 0.095, 0.001)
    rotations = [0.0, np.pi/6, np.pi/4, np.pi/3]
    
    for spacing in spacings:
        if not time_check():
            break
        for rot in rotations:
            if not time_check():
                break
            pts, r = _hex_initial(spacing, rotation=rot)
            if pts is not None and r > best_radius:
                best_radius = r
                best_points = pts.copy()
    
    if best_points is None:
        # Fallback: uniform random in circle
        rng = np.random.RandomState(0)
        angles = rng.uniform(0, 2 * np.pi, N)
        radii = np.sqrt(rng.uniform(0, 1, N))
        best_points = np.column_stack([
            radii * np.cos(angles),
            radii * np.sin(angles)
        ])
        best_radius = _compute_radius_fast(best_points)
    
    # --- Stage 2: Local optimization with time-budgeted stages ---
    # Run repulsion relaxation first (fast, fixes obvious overlaps)
    rep_budget = deadline - 60.0
    rep_pts, rep_r = _repulsion_relax(best_points, rep_budget, seed=123)
    if rep_r > best_radius:
        best_points = rep_pts
        best_radius = rep_r
    
    # Run local search with different seeds
    for seed in [42, 777, 999, 1234]:
        if not time_check():
            break
        remaining = deadline - time.time()
        if remaining < 10.0:
            break
        ls_budget = time.time() + min(remaining * 0.6, 30.0)
        ls_pts, ls_r = _local_search(best_points, ls_budget, seed=seed, use_fast=True)
        if ls_r > best_radius:
            best_points = ls_pts
            best_radius = ls_r
    
    # Run gradient ascent if time permits
    if time_check():
        remaining = deadline - time.time()
        if remaining > 15.0:
            grad_budget = time.time() + min(remaining * 0.5, 25.0)
            grad_pts, grad_r = _gradient_ascent(best_points, grad_budget, seed=456)
            if grad_r > best_radius:
                best_points = grad_pts
                best_radius = grad_r
    
    # Final high-precision polish
    if time_check():
        remaining = deadline - time.time()
        if remaining > 5.0:
            polish_budget = time.time() + min(remaining * 0.7, 10.0)
            polish_pts, polish_r = _local_search(best_points, polish_budget, seed=12345, use_fast=False)
            if polish_r > best_radius:
                best_points = polish_pts
                best_radius = polish_r
    
    # Final validation and return
    # Ensure points are strictly inside
    dists_origin = np.hypot(best_points[:, 0], best_points[:, 1])
    max_allowed = 1.0 - 1e-9
    mask = dists_origin >= max_allowed
    if np.any(mask):
        best_points[mask] *= (max_allowed / dists_origin[mask]) * 0.9999
    
    # Ensure no coincident points
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
    
    # Final check
    if not np.all(np.isfinite(best_points)) or best_points.shape != (N, 2):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = np.sqrt(np.linspace(0, 1, N))
        best_points = np.column_stack([
            radii * np.cos(angles),
            radii * np.sin(angles)
        ])
        best_points *= 0.99
    
    return best_points
