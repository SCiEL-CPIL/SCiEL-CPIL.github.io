"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def _hex_init(n, seed=0):
    """Generate n points from a hexagonal lattice inside the unit disk."""
    rng = np.random.RandomState(seed)
    for attempt in range(30):
        spacing = 0.052 + rng.random() * 0.015
        k = int(2.0 / spacing) + 5
        xs, ys = np.meshgrid(
            np.arange(-k, k + 1) * spacing,
            np.arange(-k, k + 1) * spacing * np.sqrt(3.0) / 2.0
        )
        for j in range(2 * k + 1):
            if j % 2 == 1:
                xs[j, :] += spacing / 2.0
        pts = np.column_stack([xs.ravel(), ys.ravel()])
        r = np.hypot(pts[:, 0], pts[:, 1])
        mask = r <= 1.0 - spacing / 2.0 - 0.002
        pts = pts[mask]
        if len(pts) >= n:
            idx = np.argsort(r[mask])[:n]
            return pts[idx].copy()
    # Fallback: random points
    angles = rng.uniform(0, 2 * np.pi, n)
    radii = rng.uniform(0, 0.85, n)
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _compute_radius(pts):
    """Compute the maximum radius for given points."""
    n = len(pts)
    if n < 2:
        return min(1.0 - np.max(np.hypot(pts[:, 0], pts[:, 1])), 0.5)
    
    # Use scipy's cdist or manual for pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    boundary_dists = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair_dist / 2.0, min_boundary)


def _soft_min(values, temperature=0.01):
    """Compute soft minimum of an array of values."""
    values = np.asarray(values)
    if len(values) == 0:
        return 0.0
    # Use log-sum-exp trick for stability
    min_val = np.min(values)
    shifted = values - min_val
    # soft_min ≈ min_val - temperature * log(sum(exp(-shifted/temperature)))
    # But we want to approximate min, so:
    # soft_min ≈ min_val + temperature * log(sum(exp((min_val - values)/temperature)))
    # Actually simpler: just use the min for gradient computation
    return min_val


def _gradient_ascent(pts, n, steps=500, lr_init=0.001, lr_decay=0.999, temp=0.005):
    """
    Gradient ascent on the soft-min of pairwise distances and boundary distances.
    We maximize min(d_ij/2, 1-|c_i|) by moving points to increase the minimum.
    """
    pts = pts.copy().astype(np.float64)
    lr = lr_init
    
    for step in range(steps):
        # Compute pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-12)
        np.fill_diagonal(dists, np.inf)
        
        # Boundary distances
        boundary = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
        
        # Find the critical constraints (those closest to the minimum)
        min_pair = np.min(dists)
        min_boundary = np.min(boundary)
        r_current = min(min_pair / 2.0, min_boundary)
        
        # Compute gradient: for each point, sum of unit vectors away from neighbors
        # and away from boundary
        grad = np.zeros_like(pts)
        
        # Pairwise gradient: for each pair (i,j), if d_ij is small, push apart
        # Use inverse distance weighting
        # grad_i += sum_j (c_i - c_j) / d_ij^2 for pairs where d_ij < 2*r_target
        r_target = r_current * 1.001 + 0.00005
        
        # Only consider pairs that are close to the constraint
        mask = dists < 2 * r_target + 0.01
        np.fill_diagonal(mask, False)
        
        for i in range(n):
            neighbors = np.where(mask[i])[0]
            if len(neighbors) > 0:
                for j in neighbors:
                    d = dists[i, j]
                    if d > 1e-10:
                        # Push i away from j
                        grad[i] += (pts[i] - pts[j]) / (d * d)
        
        # Boundary gradient: push points inward if close to boundary
        boundary_mask = boundary < r_target + 0.01
        for i in np.where(boundary_mask)[0]:
            r_i = np.hypot(pts[i, 0], pts[i, 1])
            if r_i > 1e-10:
                # Push toward center
                grad[i] -= (pts[i] / r_i) * (1.0 / max(r_i, 0.01))
        
        # Normalize gradient to prevent large steps
        grad_norm = np.sqrt(np.sum(grad ** 2))
        if grad_norm > 1e-10:
            grad /= grad_norm
        
        # Update
        pts += lr * grad
        
        # Keep points inside the unit disk (hard constraint)
        r_pts = np.hypot(pts[:, 0], pts[:, 1])
        mask_out = r_pts > 1.0 - 0.001
        if np.any(mask_out):
            pts[mask_out] *= (1.0 - 0.001) / r_pts[mask_out, np.newaxis]
        
        # Decay learning rate
        lr *= lr_decay
        if lr < 1e-6:
            break
    
    return pts


def _local_optimization(pts, n, time_limit=20.0):
    """
    Local optimization using a combination of gradient ascent and direct search.
    Focus on the most constrained points.
    """
    start_time = time.time()
    pts = pts.copy().astype(np.float64)
    
    # First pass: gradient ascent
    pts = _gradient_ascent(pts, n, steps=300, lr_init=0.002, lr_decay=0.998)
    
    elapsed = time.time() - start_time
    if elapsed > time_limit * 0.5:
        return pts
    
    # Second pass: try to improve the worst constraint
    best_r = _compute_radius(pts)
    
    for iteration in range(50):
        if time.time() - start_time > time_limit:
            break
        
        # Find the most constrained point(s)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        boundary = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
        
        # For each point, find its minimum distance to others and to boundary
        min_dist_to_other = np.min(dists, axis=1)  # min distance to any other point
        min_constraint = np.minimum(min_dist_to_other / 2.0, boundary)
        
        # Find the point with the smallest constraint
        worst_idx = np.argmin(min_constraint)
        worst_val = min_constraint[worst_idx]
        
        # Try to move this point to improve its constraint
        # Compute the direction that maximizes the minimum constraint
        # Simple approach: move away from the closest neighbor
        closest_neighbor = np.argmin(dists[worst_idx])
        d_closest = dists[worst_idx, closest_neighbor]
        
        if d_closest > 1e-10:
            # Move away from closest neighbor
            direction = (pts[worst_idx] - pts[closest_neighbor]) / d_closest
            # Also consider boundary
            r_i = np.hypot(pts[worst_idx, 0], pts[worst_idx, 1])
            if boundary[worst_idx] < min_dist_to_other[worst_idx] / 2.0:
                # Boundary is more constraining, move toward center
                direction = -pts[worst_idx] / max(r_i, 1e-10)
            
            # Try a small step in this direction
            step_size = 0.001
            new_pt = pts[worst_idx].copy()
            new_pt += step_size * direction
            
            # Check if it improves the constraint
            new_r_i = np.hypot(new_pt[0], new_pt[1])
            new_boundary = 1.0 - new_r_i
            new_dists = np.sqrt(np.sum((pts - new_pt[np.newaxis, :]) ** 2, axis=1))
            new_min_dist = np.min(new_dists)
            new_constraint = min(new_min_dist / 2.0, new_boundary)
            
            if new_constraint > worst_val + 1e-8:
                pts[worst_idx] = new_pt
    
    return pts


def construct():
    n = N
    deadline = time.time() + 140  # Leave 10s margin before 150s kill
    
    # Get initial hexagonal packing
    x0 = _hex_init(n, seed=42)
    best_pts = x0.copy()
    best_r = _compute_radius(best_pts)
    
    # Try multiple seeds for initialization
    for seed in [42, 123, 456, 789, 1000]:
        if time.time() > deadline - 30:
            break
        
        x_init = _hex_init(n, seed=seed)
        
        # Local optimization
        optimized = _local_optimization(x_init, n, time_limit=20.0)
        
        r_val = _compute_radius(optimized)
        if r_val > best_r:
            best_r = r_val
            best_pts = optimized.copy()
        
        if time.time() > deadline - 15:
            break
    
    # Final polish on the best result
    if time.time() < deadline - 10:
        polished = _local_optimization(best_pts, n, time_limit=10.0)
        r_val = _compute_radius(polished)
        if r_val > best_r:
            best_r = r_val
            best_pts = polished.copy()
    
    return best_pts
# EVOLVE-BLOCK-END
