"""Force-directed packing with adaptive step-size and restarts."""
import numpy as np
import time
import math

N = 300

# EVOLVE-BLOCK-START

def _hex_lattice_init():
    """Generate a hexagonal lattice of points inside the unit disk, return first N."""
    best_points = None
    for s in [0.10, 0.102, 0.104, 0.106, 0.108, 0.110, 0.112, 0.115, 0.12]:
        k = int(1.0 / s) + 3
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * math.sqrt(3.0) / 2.0
                if x * x + y * y <= 1.0:
                    points.append((x, y))
        points = np.array(points)
        if len(points) >= N:
            r = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(r, kind='stable')[:N]
            pts = points[idx]
            if best_points is None or len(pts) == N:
                best_points = pts
                if len(pts) == N:
                    break
    if best_points is not None and len(best_points) == N:
        return best_points
    
    # Fallback: very fine grid
    s = 0.05
    k = int(1.0 / s) + 3
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * math.sqrt(3.0) / 2.0
            if x * x + y * y <= 1.0:
                points.append((x, y))
    points = np.array(points)
    r = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(r, kind='stable')[:N]
    return points[idx]


def _score(points):
    """Compute the maximum radius r for given centers."""
    dist_boundary = 1.0 - np.hypot(points[:, 0], points[:, 1])
    min_boundary = np.min(dist_boundary)
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_boundary, min_pair / 2.0)


def _force_directed_optimize(points, time_limit):
    """
    Force-directed relaxation:
    - Repulsion between all pairs: F_repel = (target_dist - dist) * direction if dist < target_dist
    - Boundary force: pull points inward if too close to boundary
    - Adaptive step size
    """
    start_time = time.time()
    best_score = _score(points)
    best_points = points.copy()
    
    # Initial target distance based on current score
    target_dist = 2.0 * best_score * 1.01  # Slightly larger than current min distance
    
    # Adaptive step size
    step_size = 0.01
    min_step = 1e-6
    max_step = 0.05
    
    # Number of iterations
    max_iter = 100000
    patience = 5000
    no_improve_count = 0
    
    for iteration in range(max_iter):
        # Check time
        if time.time() - start_time > time_limit * 0.9:
            break
        
        pts = best_points.copy()
        
        # Compute forces
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Only consider pairs closer than target_dist
        mask = dists < target_dist
        # For efficiency, we can limit to nearest neighbors, but for 300 points it's OK
        # Compute force for all pairs (vectorized)
        # Force magnitude: (target_dist - dist) / target_dist
        # Direction: (pts_i - pts_j) / dist
        # F_i += sum_j (target_dist - dist_ij) * (pts_i - pts_j) / (dist_ij * target_dist)
        
        # To avoid O(N^2) memory issues, we do it in a loop over i
        # But 300 points is small enough
        inv_dists = np.zeros_like(dists)
        valid = dists > 1e-10
        inv_dists[valid] = 1.0 / dists[valid]
        
        # Force contribution: (target_dist - dist) * direction / target_dist
        # = (target_dist - dist) * (pts_i - pts_j) / (dist * target_dist)
        # For each i, sum over j != i
        # F_i = sum_j (target_dist - dist_ij) * (pts_i - pts_j) / (dist_ij * target_dist)
        #     = (1/target_dist) * sum_j (target_dist - dist_ij) * (pts_i - pts_j) / dist_ij
        #     = (1/target_dist) * [ target_dist * sum_j (pts_i - pts_j)/dist_ij - sum_j (pts_i - pts_j) ]
        # This is getting complex. Let's just compute directly.
        
        # Direct computation:
        # F_i = sum_{j!=i} (target_dist - dist_ij) * (pts_i - pts_j) / (dist_ij * target_dist)
        #      = (1/target_dist) * sum_{j!=i} (target_dist - dist_ij) * (pts_i - pts_j) / dist_ij
        
        # Let's compute this vectorized:
        # diff[i,j] = pts[i] - pts[j]
        # dist[i,j] = |diff[i,j]|
        # weight[i,j] = (target_dist - dist[i,j]) / (dist[i,j] * target_dist) if dist < target_dist else 0
        # F_i = sum_j weight[i,j] * diff[i,j]
        
        weight = np.zeros_like(dists)
        valid_mask = (dists < target_dist) & (dists > 1e-10)
        weight[valid_mask] = (target_dist - dists[valid_mask]) / (dists[valid_mask] * target_dist)
        
        # F_i = sum_j weight[i,j] * diff[i,j]
        # diff shape: (N, N, 2), weight shape: (N, N)
        forces_repel = np.sum(weight[:, :, np.newaxis] * diff, axis=1)
        
        # Boundary force: pull inward if |pts| > 1 - margin
        margin = 0.05
        r = np.hypot(pts[:, 0], pts[:, 1])
        boundary_mask = r > (1.0 - margin)
        if np.any(boundary_mask):
            # Force magnitude proportional to how far outside the safe zone
            excess = r[boundary_mask] - (1.0 - margin)
            # Direction: inward (towards origin)
            inward = -pts[boundary_mask] / r[boundary_mask, np.newaxis]
            forces[boundary_mask] += excess * inward * 10.0  # Scale factor
        
        forces += forces_repel
        
        # Update positions
        new_pts = pts + step_size * forces
        
        # Ensure points stay inside the unit disk
        new_r = np.hypot(new_pts[:, 0], new_pts[:, 1])
        outside = new_r > 1.0
        if np.any(outside):
            new_pts[outside] = new_pts[outside] / new_r[outside, np.newaxis] * (1.0 - 1e-10)
        
        # Compute new score
        new_score = _score(new_pts)
        
        if new_score > best_score + 1e-12:
            best_score = new_score
            best_points = new_pts.copy()
            no_improve_count = 0
            # Increase step size slightly
            step_size = min(step_size * 1.05, max_step)
            # Increase target distance slightly
            target_dist = max(target_dist * 1.001, 2.0 * best_score * 1.01)
        else:
            no_improve_count += 1
            # Decrease step size
            step_size = max(step_size * 0.95, min_step)
            if no_improve_count > patience:
                break
    
    return best_points


def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin from 120s
    
    # Generate initial hexagonal lattice
    points = _hex_lattice_init()
    if points is None or len(points) != N:
        # Emergency fallback
        points = np.zeros((N, 2))
        for i in range(N):
            angle = 2 * np.pi * i / N
            r = 0.5
            points[i, 0] = r * np.cos(angle)
            points[i, 1] = r * np.sin(angle)
    
    init_score = _score(points)
    
    # Run force-directed optimization
    elapsed = time.time() - start_time
    if elapsed < time_limit * 0.8:
        remaining = time_limit - elapsed - 5.0  # 5s safety
        if remaining > 10:
            points = _force_directed_optimize(points, remaining)
    
    # Final check: ensure validity
    final_score = _score(points)
    
    # Ensure all points are inside the unit disk
    r = np.hypot(points[:, 0], points[:, 1])
    if np.any(r > 1.0):
        mask = r > 1.0
        points[mask] = points[mask] / r[mask, np.newaxis] * (1.0 - 1e-10)
    
    # Ensure no coincident points
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-10:
        for i in range(N):
            for j in range(i + 1, N):
                if np.linalg.norm(points[i] - points[j]) < 1e-10:
                    points[j, 0] += 1e-6
    
    return points

# EVOLVE-BLOCK-END
