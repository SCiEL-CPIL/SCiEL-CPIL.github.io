"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_score(points):
    """Compute the minimum radius for a set of points."""
    n = len(points)
    if n == 0:
        return 0.0
    dists = np.sqrt(np.sum(points**2, axis=1))
    boundary_r = np.min(1.0 - dists)
    if boundary_r < 0:
        return 0.0
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair_dist = np.min(dist_matrix)
    pair_r = min_pair_dist / 2.0
    return min(boundary_r, pair_r)


def _fibonacci_disk_init(n, seed=42):
    """Generate n points in the unit disk using a Fibonacci-like spiral."""
    rng = np.random.RandomState(seed)
    # Use golden angle spiral
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    theta = golden_angle * np.arange(n)
    # Radial distribution: uniform in area means r ~ sqrt(uniform)
    # But for better packing, use a more even distribution
    r = np.sqrt((np.arange(n) + 0.5) / n)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    points = np.column_stack([x, y])
    # Add small random perturbation to break symmetry
    points += rng.normal(0, 0.01, size=points.shape)
    # Ensure all points are inside the unit disk
    radii = np.sqrt(np.sum(points**2, axis=1))
    mask = radii < 1.0
    if not np.all(mask):
        points[~mask] *= (1.0 - 0.01) / radii[~mask, np.newaxis]
    return points


def _force_relaxation(points, max_time=100.0, initial_lr=0.001, min_lr=1e-6):
    """Relax points using repulsion forces to maximize minimum distance."""
    n = len(points)
    points = points.copy().astype(np.float64)
    best_points = points.copy()
    best_score = _compute_score(points)
    
    start_time = time.time()
    lr = initial_lr
    temp = 0.1  # Softness parameter for smooth min
    
    while time.time() - start_time < max_time:
        # Compute pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist_matrix, np.inf)
        
        # Compute forces: repulsion from all other points
        forces = np.zeros_like(points)
        
        # Pairwise repulsion force: F_ij = (p_i - p_j) / |p_i - p_j|^2 for close pairs
        # Use a smooth approximation
        # Only consider pairs within some cutoff for efficiency
        cutoff = 0.15
        mask = dist_matrix < cutoff
        np.fill_diagonal(mask, False)
        
        if np.any(mask):
            # For each point, sum up repulsion from neighbors
            for i in range(n):
                neighbors = np.where(mask[i])[0]
                if len(neighbors) > 0:
                    for j in neighbors:
                        d = dist_matrix[i, j]
                        if d > 1e-10:
                            # Repulsive force proportional to 1/d^2
                            force_mag = 1.0 / (d * d)
                            forces[i] += (points[i] - points[j]) / d * force_mag
        
        # Boundary force: push points inward if too close to boundary
        radii = np.sqrt(np.sum(points**2, axis=1))
        boundary_margin = 0.1  # Start pushing when within this margin of boundary
        for i in range(n):
            if radii[i] > 1.0 - boundary_margin:
                # Force proportional to how close to boundary
                excess = radii[i] - (1.0 - boundary_margin)
                if radii[i] > 1e-10:
                    forces[i] -= points[i] / radii[i] * excess * 10.0
        
        # Update positions
        points += lr * forces
        
        # Clamp to unit disk
        radii = np.sqrt(np.sum(points**2, axis=1))
        mask = radii > 1.0
        if np.any(mask):
            points[mask] *= (1.0 - 1e-6) / radii[mask, np.newaxis]
        
        # Compute score
        score = _compute_score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
        
        # Learning rate schedule
        lr *= 0.999
        if lr < min_lr:
            lr = min_lr
        
        # If stuck, try a small random perturbation
        if time.time() - start_time > 10:
            elapsed = time.time() - start_time
            if elapsed > 50 and score < best_score:
                # Add small noise to escape local minima
                points += np.random.normal(0, 0.001, size=points.shape)
                radii = np.sqrt(np.sum(points**2, axis=1))
                mask = radii > 1.0
                if np.any(mask):
                    points[mask] *= (1.0 - 1e-6) / radii[mask, np.newaxis]
    
    return best_points, best_score


def construct():
    """Construct 300 circle centers in the unit disk."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Initialize with Fibonacci disk
    points = _fibonacci_disk_init(N, seed=42)
    
    # Phase 1: Coarse relaxation
    points, score = _force_relaxation(points, max_time=60.0, initial_lr=0.002, min_lr=1e-5)
    
    if time.time() > deadline:
        return points
    
    # Phase 2: Fine relaxation with smaller learning rate
    points, score2 = _force_relaxation(points, max_time=40.0, initial_lr=0.0005, min_lr=1e-7)
    if score2 > score:
        points = points
        score = score2
    
    # Final check: ensure all points are strictly inside
    radii = np.sqrt(np.sum(points**2, axis=1))
    mask = radii >= 1.0
    if np.any(mask):
        points[mask] *= 0.999 / radii[mask, np.newaxis]
    
    return points

# EVOLVE-BLOCK-END
