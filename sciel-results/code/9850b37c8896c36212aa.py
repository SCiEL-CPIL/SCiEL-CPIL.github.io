"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _compute_radius(centers):
    """Compute the valid radius for a set of centers inside the unit circle."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Distance to boundary
    dists_boundary = 1.0 - np.hypot(centers[:, 0], centers[:, 1])
    r_boundary = np.min(dists_boundary)
    if r_boundary <= 0:
        return 0.0
    # Pairwise distances
    x = centers[:, 0]
    y = centers[:, 1]
    dx = np.abs(x[:, None] - x[None, :])
    dy = np.abs(y[:, None] - y[None, :])
    dists = np.hypot(dx, dy)
    np.fill_diagonal(dists, np.inf)
    r_pair = np.min(dists) / 2.0
    r = min(r_boundary, r_pair)
    if r <= 0:
        return 0.0
    return r

def _hexagonal_packing():
    """Generate a hexagonal lattice packing inside the unit circle."""
    # Try different spacings to find a good packing
    best_points = None
    best_radius = 0.0
    
    for spacing in np.linspace(0.15, 0.25, 20):
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        radius = radius[mask]
        
        if len(points) >= N:
            # Select the N points closest to the center
            points = points[np.argsort(radius, kind='stable')[:N]]
            r = _compute_radius(points)
            if r > best_radius:
                best_radius = r
                best_points = points.copy()
    
    if best_points is not None:
        return best_points
    
    # Fallback: simple grid
    spacing = 0.2
    k = int(1.0 / spacing) + 2
    rows = [(i * spacing, j * spacing) for i in range(-k, k + 1) for j in range(-k, k + 1)]
    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    mask = radius <= 1.0 - spacing / 2.0
    points = points[mask]
    radius = radius[mask]
    if len(points) >= N:
        return points[np.argsort(radius, kind='stable')[:N]]
    
    # Last resort: evenly spaced on a circle
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([0.5 * np.cos(angles), 0.5 * np.sin(angles)])

def _anneal(centers, max_iters=5000, dt=0.01):
    """Simulated annealing to improve the packing."""
    best_centers = centers.copy()
    best_radius = _compute_radius(centers)
    current_radius = best_radius
    temp = 0.05
    
    for it in range(max_iters):
        # Generate a random perturbation
        new_centers = centers.copy()
        idx = np.random.randint(N)
        angle = np.random.uniform(0, 2 * np.pi)
        dist = np.random.uniform(0, dt)
        new_centers[idx, 0] += dist * np.cos(angle)
        new_centers[idx, 1] += dist * np.sin(angle)
        
        new_radius = _compute_radius(new_centers)
        
        if new_radius > current_radius:
            centers = new_centers
            current_radius = new_radius
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = centers.copy()
        else:
            # Accept with probability exp((new - current) / temp)
            prob = np.exp((new_radius - current_radius) / max(temp, 1e-10))
            if np.random.random() < prob:
                centers = new_centers
                current_radius = new_radius
        
        # Cool down
        temp *= 0.999
    
    return best_centers, best_radius

def construct():
    """Construct the packing of N equal circles inside the unit circle."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with hexagonal packing
    centers = _hexagonal_packing()
    best_centers = centers.copy()
    best_radius = _compute_radius(centers)
    
    # Run annealing with multiple restarts if time permits
    num_restarts = 0
    while time.time() < deadline:
        # Anneal from current best
        improved_centers, improved_radius = _anneal(
            best_centers if num_restarts == 0 else centers,
            max_iters=2000,
            dt=0.005
        )
        
        if improved_radius > best_radius:
            best_radius = improved_radius
            best_centers = improved_centers.copy()
        
        # Check if we should do another restart
        if time.time() > deadline - 30.0:
            break
        
        # Perturb for next restart
        centers = best_centers + np.random.normal(0, 0.01, (N, 2))
        num_restarts += 1
    
    # Final validation
    final_radius = _compute_radius(best_centers)
    if final_radius > 0:
        return best_centers
    
    # Fallback if something went wrong
    return _hexagonal_packing()
# EVOLVE-BLOCK-END
