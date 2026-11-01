"""Improved packing: force-directed relaxation with adaptive time-stepping."""
import numpy as np
import time

N = 50


def _compute_radius(points):
    """Compute the maximum valid radius for given centre positions."""
    # Boundary constraint
    dists = np.hypot(points[:, 0], points[:, 1])
    boundary_r = np.min(1.0 - dists)
    
    # Pairwise constraint
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists2 = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists2, np.inf)
    min_pair_dist = np.min(dists2)
    pair_r = min_pair_dist / 2.0
    
    return min(boundary_r, pair_r)


def _force_directed(points, max_iters=5000, dt_init=0.01, dt_min=1e-6, dt_decay=0.999):
    """Force-directed relaxation to maximize the minimum radius."""
    points = points.copy().astype(np.float64)
    n = len(points)
    
    # We want to maximize r = min(min_pair_dist/2, min(1-|c_i|))
    # Use a repulsive force model:
    # - Pairs that are too close push apart
    # - Points too close to boundary get pulled inward
    
    best_r = _compute_radius(points)
    best_points = points.copy()
    
    dt = dt_init
    
    for it in range(max_iters):
        forces = np.zeros_like(points)
        
        # Pairwise repulsion: push apart points that are closer than target
        # Target spacing: we want all pairs to be at least 2*r apart
        # Use a soft repulsion that increases as distance decreases
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dists2, 1.0)
        dists = np.sqrt(dists2)
        
        # Repulsive force: k / dist^2 in direction away from other point
        # Only for pairs that are relatively close
        # Use a threshold-based approach
        threshold = 2.0 * max(best_r, 0.05)
        mask = (dists < threshold) & (dists > 1e-10)
        
        # Force magnitude: proportional to (threshold - dist) for dist < threshold
        force_mag = np.zeros_like(dists)
        force_mag[mask] = (threshold - dists[mask]) / (dists[mask] + 1e-10)
        
        # Sum forces from each pair
        for i in range(n):
            for j in range(i + 1, n):
                if mask[i, j]:
                    f = force_mag[i, j]
                    direction = diff[i, j] / (dists[i, j] + 1e-10)
                    forces[i] += f * direction
                    forces[j] -= f * direction
        
        # Boundary force: pull points inward if they're too close to boundary
        dists_center = np.hypot(points[:, 0], points[:, 1])
        boundary_margin = 1.0 - dists_center
        # Target margin: at least best_r from boundary
        boundary_force_mag = np.zeros(n)
        for i in range(n):
            if boundary_margin[i] < best_r:
                # Pull toward center
                boundary_force_mag[i] = (best_r - boundary_margin[i]) / (dists_center[i] + 1e-10)
        
        forces[:, 0] += boundary_force_mag * points[:, 0] / (dists_center + 1e-10)
        forces[:, 1] += boundary_force_mag * points[:, 1] / (dists_center + 1e-10)
        
        # Update positions
        points += dt * forces
        
        # Project back: ensure points stay inside unit circle
        dists_new = np.hypot(points[:, 0], points[:, 1])
        outside = dists_new > 1.0
        if np.any(outside):
            points[outside, 0] /= dists_new[outside, None]
            points[outside, 1] /= dists_new[outside, None]
            # Add small inward nudge
            points[outside, 0] *= 0.999
            points[outside, 1] *= 0.999
        
        # Compute current radius
        current_r = _compute_radius(points)
        if current_r > best_r:
            best_r = current_r
            best_points = points.copy()
        
        # Adaptive time step: decrease if radius decreased significantly
        if current_r < best_r * 0.99:
            dt *= 0.95
        else:
            dt *= dt_decay
        
        dt = max(dt, dt_min)
        
        # Convergence check
        if it > 100 and abs(current_r - best_r) < 1e-10:
            break
    
    return best_points, best_r


def _refine_with_gradient(points, max_iters=2000, lr_init=0.001, lr_decay=0.999):
    """Gradient-based refinement using subgradient of the min function."""
    points = points.copy().astype(np.float64)
    n = len(points)
    
    best_r = _compute_radius(points)
    best_points = points.copy()
    lr = lr_init
    
    for it in range(max_iters):
        # Compute which constraint is active
        dists = np.hypot(points[:, 0], points[:, 1])
        boundary_r = 1.0 - np.min(dists)
        
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists2 = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists2, np.inf)
        min_pair_dist = np.min(dists2)
        pair_r = min_pair_dist / 2.0
        
        r = min(boundary_r, pair_r)
        
        if r > best_r:
            best_r = r
            best_points = points.copy()
        
        # Compute subgradient
        grad = np.zeros_like(points)
        
        if pair_r <= boundary_r:
            # Pairwise constraint is active
            # Find the pair(s) achieving min distance
            min_pairs = np.argwhere(np.isclose(dists2, min_pair_dist, atol=1e-10))
            for i, j in min_pairs:
                if i == j:
                    continue
                direction = (points[i] - points[j]) / (min_pair_dist + 1e-10)
                # Gradient of dist(i,j)/2 w.r.t. point i is direction/2
                grad[i] += direction / 2.0
                grad[j] -= direction / 2.0
        else:
            # Boundary constraint is active
            min_idx = np.argmin(dists)
            direction = points[min_idx] / (dists[min_idx] + 1e-10)
            grad[min_idx] -= direction
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad /= grad_norm
        
        # Update
        points += lr * grad
        
        # Project back inside unit circle
        dists_new = np.hypot(points[:, 0], points[:, 1])
        outside = dists_new > 1.0
        if np.any(outside):
            points[outside, 0] /= dists_new[outside, None]
            points[outside, 1] /= dists_new[outside, None]
            points[outside, 0] *= 0.999
            points[outside, 1] *= 0.999
        
        lr *= lr_decay
        if lr < 1e-8:
            break
    
    return best_points, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial configuration: hexagonal lattice
    def make_hex_initial():
        spacing = 0.28
        while True:
            k = int(1.0 / spacing) + 2
            rows = []
            for j in range(-k, k + 1):
                for i in range(-k, k + 1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * np.sqrt(3.0) / 2.0
                    if x * x + y * y <= (1.0 - spacing / 2.0) ** 2:
                        rows.append([x, y])
            points = np.array(rows)
            if len(points) >= N:
                # Select N points closest to center
                radii = np.hypot(points[:, 0], points[:, 1])
                idx = np.argsort(radii)[:N]
                return points[idx]
            spacing *= 0.95
    
    best_points = make_hex_initial()
    best_r = _compute_radius(best_points)
    
    # Force-directed relaxation
    if time.time() < deadline - 30:
        points_fd, r_fd = _force_directed(best_points, max_iters=3000)
        if r_fd > best_r:
            best_r = r_fd
            best_points = points_fd
    
    # Gradient refinement
    if time.time() < deadline - 15:
        points_gr, r_gr = _refine_with_gradient(best_points, max_iters=3000)
        if r_gr > best_r:
            best_r = r_gr
            best_points = points_gr
    
    # Final verification and cleanup
    # Ensure all points are strictly inside the unit circle
    dists = np.hypot(best_points[:, 0], best_points[:, 1])
    for i in range(N):
        if dists[i] >= 1.0:
            best_points[i, 0] *= 0.999
            best_points[i, 1] *= 0.999
    
    # Ensure no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-10:
                best_points[j, 0] += 1e-8
    
    return best_points
