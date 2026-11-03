"""Rewritten program for packing 300 equal circles in a unit circle."""
import numpy as np
import time

N = 300


def _compute_score(points):
    """Compute the largest common radius for given centres."""
    if len(points) != N:
        return 0.0
    # Check for non-finite values
    if not np.all(np.isfinite(points)):
        return 0.0
    # Check centres are inside or on the container
    dists = np.sqrt(np.sum(points**2, axis=1))
    if np.any(dists > 1.0 + 1e-12):
        return 0.0
    # Compute minimum distance between any two centres
    # Use a spatial approach for efficiency
    min_dist = np.inf
    # For 300 points, O(N^2) is 90000 pairs, which is fast
    for i in range(N):
        diff = points[i+1:] - points[i]
        d2 = np.sum(diff**2, axis=1)
        if len(d2) > 0:
            md = np.min(d2)
            if md < min_dist:
                min_dist = md
    min_dist = np.sqrt(min_dist)
    # Check for coincident centres
    if min_dist < 1e-15:
        return 0.0
    r = min(min_dist / 2.0, 1.0 - np.max(dists))
    return r


def _hex_lattice_init(n_points, seed=0):
    """Generate a hexagonal lattice initialization."""
    rng = np.random.default_rng(seed)
    # Start with a hexagonal lattice
    spacing = 2.0 / (np.sqrt(n_points) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        if len(points) >= n_points:
            # Add small random perturbation to break symmetry
            points = points + rng.uniform(-0.01 * spacing, 0.01 * spacing, size=points.shape)
            return points[:n_points]
        spacing *= 0.95


def _random_init(n_points, seed=0):
    """Generate a random initialization inside the unit disk."""
    rng = np.random.default_rng(seed)
    while True:
        angles = rng.uniform(0, 2 * np.pi, n_points)
        radii = np.sqrt(rng.uniform(0, 1.0, n_points))
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        # Check if no two points are too close
        min_dist = np.inf
        for i in range(n_points):
            diff = points[i+1:] - points[i]
            d2 = np.sum(diff**2, axis=1)
            if len(d2) > 0:
                md = np.min(d2)
                if md < min_dist:
                    min_dist = md
        min_dist = np.sqrt(min_dist)
        if min_dist > 0.01:
            return points


def _local_optimize(points, max_iter=2000, tol=1e-10, deadline=None):
    """Local optimization using gradient-free method (Nelder-Mead style for packing).
    
    Uses a simple iterative repulsion/attraction scheme.
    """
    points = points.copy()
    n = len(points)
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute forces
        forces = np.zeros_like(points)
        min_dist = np.inf
        
        # Pairwise repulsion
        for i in range(n):
            diff = points[i+1:] - points[i]
            d2 = np.sum(diff**2, axis=1)
            d = np.sqrt(d2)
            # Avoid division by zero
            d = np.maximum(d, 1e-10)
            
            # Repulsive force: push apart if too close
            # Target distance is related to optimal spacing
            target = 0.1  # Approximate optimal spacing for 300 circles
            # Force is proportional to how much closer than target
            excess = np.maximum(target - d, 0)
            force_mag = excess / (d + 1e-10)
            
            # Only repel if closer than target
            mask = d < target
            if np.any(mask):
                f = force_mag[mask] * (diff[mask] / d[mask, None])
                forces[i] -= np.sum(f, axis=0)
                forces[i+1:][mask] += f
            
            if np.any(d < min_dist):
                min_dist = np.min(d)
        
        # Attraction to stay inside container (radial force)
        dists = np.sqrt(np.sum(points**2, axis=1))
        # If close to boundary, push inward
        boundary_excess = np.maximum(dists - 0.95, 0)
        radial_force = boundary_excess * (points / np.maximum(dists, 1e-10)[:, None])
        forces -= radial_force
        
        # Update positions with adaptive step size
        force_mags = np.sqrt(np.sum(forces**2, axis=1))
        if np.max(force_mags) < tol:
            break
        
        # Adaptive step size
        step = 0.01 / (1 + iteration * 0.001)
        points += step * forces
        
        # Project back inside container
        new_dists = np.sqrt(np.sum(points**2, axis=1))
        outside = new_dists > 1.0
        if np.any(outside):
            points[outside] *= (1.0 / new_dists[outside])[:, None]
    
    return points


def _pairwise_optimize(points, max_iter=500, deadline=None):
    """Optimize by iteratively adjusting pairs that are too close."""
    points = points.copy()
    n = len(points)
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        # Find closest pair
        min_dist = np.inf
        min_i, min_j = 0, 1
        for i in range(n):
            diff = points[i+1:] - points[i]
            d2 = np.sum(diff**2, axis=1)
            if len(d2) > 0:
                idx = np.argmin(d2)
                d = np.sqrt(d2[idx])
                if d < min_dist:
                    min_dist = d
                    min_i = i
                    min_j = i + 1 + idx
        
        if min_dist > 0.1:  # Good enough
            break
        
        # Move the two points apart
        diff = points[min_j] - points[min_i]
        d = np.linalg.norm(diff)
        if d < 1e-10:
            # Random direction
            angle = np.random.uniform(0, 2 * np.pi)
            diff = np.array([np.cos(angle), np.sin(angle)])
            d = 1.0
        
        direction = diff / d
        # Move each point half the needed distance
        needed = 0.1 - d
        if needed > 0:
            points[min_i] -= 0.5 * needed * direction
            points[min_j] += 0.5 * needed * direction
            improved = True
        
        # Project back inside container
        dists = np.sqrt(np.sum(points**2, axis=1))
        outside = dists > 1.0
        if np.any(outside):
            points[outside] *= (1.0 / dists[outside])[:, None]
        
        if not improved and iteration > 100:
            break
    
    return points


def construct():
    """Construct centres of 300 equal circles inside the unit circle."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    best_points = None
    best_score = 0.0
    
    # Try multiple initializations
    seeds = [0, 1, 2, 3, 4]
    
    for seed in seeds:
        if time.time() > deadline - 30:
            break
        
        # Hexagonal lattice init
        points = _hex_lattice_init(N, seed=seed)
        
        # Local optimization
        points = _local_optimize(points, max_iter=500, deadline=deadline - 20)
        
        # Pairwise optimization
        points = _pairwise_optimize(points, max_iter=200, deadline=deadline - 10)
        
        score = _compute_score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # Final polish with more iterations
    if best_points is not None and time.time() < deadline - 5:
        best_points = _local_optimize(best_points, max_iter=1000, deadline=deadline - 2)
        best_points = _pairwise_optimize(best_points, max_iter=300, deadline=deadline - 1)
        final_score = _compute_score(best_points)
        if final_score > best_score:
            best_score = final_score
    
    if best_points is None:
        # Fallback: simple hexagonal lattice
        points = _hex_lattice_init(N, seed=0)
        best_points = points
    
    return best_points
