"""Optimized circle packing: hex lattice init + vectorized local optimization."""
import numpy as np
import time

N = 300


def _hex_lattice_init(n):
    """Generate a hexagonal lattice packing of n circles in unit disk."""
    # We want to find a spacing such that we get at least n points inside the disk
    # with margin for radius
    for spacing in np.linspace(0.055, 0.12, 200):
        # Generate hex lattice points
        pts = []
        # Number of rows/cols needed
        extent = 2.0 / spacing + 1
        k = int(extent) + 2
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= 1.0 - spacing / 2.0:
                    pts.append((x, y))
        pts = np.array(pts)
        if len(pts) >= n:
            # Sort by radius to prefer center, take first n
            radii = np.hypot(pts[:, 0], pts[:, 1])
            idx = np.argsort(radii, kind='stable')
            return pts[idx[:n]].copy()
    # Fallback: uniform random
    return np.random.randn(n, 2) * 0.5


def _compute_score(points):
    """Compute the packing score: min of (half min pairwise dist, min boundary margin)."""
    n = len(points)
    # Pairwise distances - vectorized
    # Use chunked computation to avoid memory issues for N=300
    min_dist = np.inf
    chunk = 50
    for i in range(0, n, chunk):
        j_end = min(i + chunk, n)
        # distances from points[i:j_end] to all points
        diff = points[i:j_end, None, :] - points[None, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        # Zero out self-distances
        rows = np.arange(j_end - i)
        cols = np.arange(i, j_end)
        dists[rows, cols] = np.inf
        # Also zero out the symmetric part we already counted (lower triangle)
        # Actually just take min of upper triangle portion
        for r in range(j_end - i):
            dists[r, r] = np.inf
        min_dist = min(min_dist, np.min(dists))
    
    # Boundary margin
    radii = np.sqrt(np.sum(points * points, axis=1))
    boundary = np.min(1.0 - radii)
    
    return min(min_dist / 2.0, boundary)


def _local_optimize(points, max_iters=500, lr_init=0.01, deadline=None):
    """Local optimization using gradient-like ascent on the smooth surrogate."""
    points = points.copy()
    n = len(points)
    best_score = _compute_score(points)
    
    # Use a smooth surrogate: soft-min via log-sum-exp for the objective
    # f = softmin(min_pairwise_dist/2, boundary_margins)
    # We'll use a simpler approach: iterative repulsion + boundary attraction
    
    lr = lr_init
    t0 = time.time()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute forces
        forces = np.zeros_like(points)
        
        # Pairwise repulsion: push apart close pairs
        # Vectorized: compute all pairwise differences
        diff = points[:, None, :] - points[None, :, :]  # (n, n, 2)
        dists = np.sqrt(np.sum(diff * diff, axis=2)) + 1e-12  # (n, n)
        
        # Force proportional to 1/dist^2 for close pairs, scaled
        # Only consider pairs within some cutoff for efficiency
        cutoff = 0.15
        mask = (dists < cutoff) & (dists > 1e-10)
        
        # Force magnitude: strong repulsion for very close pairs
        inv_dist_sq = np.zeros_like(dists)
        inv_dist_sq[mask] = 1.0 / (dists[mask] ** 2)
        
        # Direction: normalize diff
        dists_safe = np.where(dists > 1e-10, dists, 1.0)
        diff_norm = diff / dists_safe[:, :, None]
        
        # Force on each point from all others
        forces += np.sum(diff_norm * inv_dist_sq[:, :, None], axis=1)
        
        # Boundary force: push inward if too close to boundary
        radii = np.sqrt(np.sum(points * points, axis=1))
        margin = 1.0 - radii
        # If margin is small, push inward strongly
        boundary_force = np.zeros_like(points)
        close_to_boundary = margin < 0.05
        if np.any(close_to_boundary):
            # Direction toward center
            radii_safe = np.where(radii > 1e-10, radii, 1.0)
            boundary_force[close_to_boundary] = -points[close_to_boundary] / radii_safe[close_to_boundary, None] * (1.0 - margin[close_to_boundary] / 0.05) ** 2 * 2.0
        
        forces += boundary_force
        
        # Update
        points += lr * forces
        
        # Keep points inside unit disk (hard constraint)
        radii = np.sqrt(np.sum(points * points, axis=1))
        too_far = radii > 0.9999
        if np.any(too_far):
            points[too_far] *= 0.9999 / radii[too_far, None]
        
        # Compute score
        score = _compute_score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
            lr = min(lr * 1.05, 0.05)
        else:
            lr = max(lr * 0.95, 0.0001)
        
        # Annealing
        lr *= 0.999
    
    return best_points, best_score


def construct():
    """Construct 300 circle centers in unit disk maximizing common radius."""
    t_start = time.time()
    deadline = t_start + 110.0  # Safety margin before 120s limit
    
    # Initialize with hex lattice
    points = _hex_lattice_init(N)
    best_score = _compute_score(points)
    best_points = points.copy()
    
    # Multiple local optimization passes with different initial perturbations
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            pts = best_points.copy()
        else:
            # Add small random perturbation
            pts = best_points + np.random.randn(N, 2) * 0.005
            # Ensure still inside
            radii = np.sqrt(np.sum(pts * pts, axis=1))
            too_far = radii > 0.99
            if np.any(too_far):
                pts[too_far] *= 0.99 / radii[too_far, None]
        
        # Local optimize
        opt_pts, opt_score = _local_optimize(
            pts, 
            max_iters=300, 
            lr_init=0.01,
            deadline=deadline
        )
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_pts.copy()
    
    # Final polish: more iterations with smaller learning rate
    if time.time() < deadline - 10:
        opt_pts, opt_score = _local_optimize(
            best_points,
            max_iters=200,
            lr_init=0.001,
            deadline=deadline
        )
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_pts.copy()
    
    # Final validation: ensure all points are strictly inside
    radii = np.sqrt(np.sum(best_points * best_points, axis=1))
    too_far = radii >= 1.0
    if np.any(too_far):
        best_points[too_far] *= 0.99999 / radii[too_far, None]
    
    return best_points
