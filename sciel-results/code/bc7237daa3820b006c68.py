"""Improved packing: hexagonal lattice initialization + local force-directed relaxation."""
import numpy as np
from scipy.spatial import cKDTree
import time

N = 300
DEADLINE = 140.0  # seconds, safety margin before 150s kill

def construct():
    start_time = time.time()
    
    # --- Step 1: Generate hexagonal lattice points inside unit disk ---
    # We need 300 points. Estimate spacing.
    # Area of unit disk = pi. 300 circles of radius r. 
    # Hex packing density ~ pi/(2*sqrt(3)) ~ 0.9069
    # 300 * pi * r^2 <= pi * 0.9069  =>  r^2 <= 0.9069/300  => r <= sqrt(0.003023) ~ 0.055
    # Known best is ~0.05315.
    # Let's generate a dense hex lattice and pick the best 300.
    
    # Generate hex lattice points
    # Spacing s. Points at distance >= s/2 from boundary.
    # We want roughly 300 points.
    # Number of points ~ area / (s^2 * sqrt(3)/2) = pi / (s^2 * sqrt(3)/2)
    # 300 ~ pi / (s^2 * sqrt(3)/2)  => s^2 ~ pi * 2 / (300 * sqrt(3)) ~ 0.01208 => s ~ 0.1099
    
    s = 0.11
    points = _hex_lattice(s, 1.0)
    
    # If we have too many, pick the 300 closest to center? Or just take first 300?
    # Better: sort by distance from center, take closest 300.
    if len(points) > N:
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:N]
        points = points[idx]
    elif len(points) < N:
        # Generate more points with smaller spacing
        s = 0.05
        points = _hex_lattice(s, 1.0)
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:N]
        points = points[idx]
    
    # Ensure exactly N points
    if len(points) > N:
        points = points[:N]
    
    # --- Step 2: Local force-directed relaxation ---
    # Maximize min distance. Use repulsive forces.
    # Objective: maximize r where r = min( min_{i<j} |ci-cj|/2, min_i (1-|ci|) )
    
    best_points = points.copy()
    best_radius = _compute_radius(points)
    
    # Relaxation loop
    dt = 0.001
    cooling = 1.0
    iterations = 5000
    time_limit = DEADLINE - (time.time() - start_time) - 5.0  # 5s margin
    
    for it in range(iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Compute pairwise distances
        tree = cKDTree(points)
        pairs = tree.query_pairs(2.0 * best_radius + 0.1)  # only close pairs
        
        # Compute forces
        forces = np.zeros_like(points)
        min_dist = float('inf')
        
        for (i, j) in pairs:
            d = points[i] - points[j]
            dist = np.linalg.norm(d)
            if dist < 1e-12:
                # Perturb
                d = np.random.randn(2) * 1e-6
                dist = np.linalg.norm(d)
            min_dist = min(min_dist, dist)
            # Repulsive force: stronger when closer
            f = d / (dist ** 3) * (1.0 if dist < best_radius * 2 else 0.1)
            forces[i] += f
            forces[j] -= f
        
        # Boundary force: push points inward if too close to boundary
        dists_center = np.linalg.norm(points, axis=1)
        for i in range(N):
            if dists_center[i] > 1.0 - best_radius:
                d = points[i]
                dist = np.linalg.norm(d)
                if dist > 1e-12:
                    forces[i] -= d / dist * (dists_center[i] - (1.0 - best_radius)) * 10.0
        
        # Update positions
        points = points + dt * cooling * forces
        cooling *= 0.995
        dt *= 0.999
        
        # Clamp to unit disk
        dists = np.linalg.norm(points, axis=1)
        mask = dists > 1.0
        if np.any(mask):
            points[mask] = points[mask] / dists[mask, None] * (1.0 - 1e-9)
        
        # Check radius
        current_radius = _compute_radius(points)
        if current_radius > best_radius:
            best_radius = current_radius
            best_points = points.copy()
    
    return best_points


def _hex_lattice(spacing, radius):
    """Generate hexagonal lattice points inside a circle of given radius."""
    points = []
    sqrt3 = np.sqrt(3.0)
    # Hex lattice: rows offset
    row_spacing = spacing * sqrt3 / 2.0
    max_row = int(radius / row_spacing) + 2
    
    for j in range(-max_row, max_row + 1):
        y = j * row_spacing
        offset = (j % 2) * spacing / 2.0
        max_i = int(radius / spacing) + 2
        for i in range(-max_i, max_i + 1):
            x = i * spacing + offset
            if x * x + y * y <= radius * radius:
                points.append([x, y])
    
    return np.array(points) if points else np.zeros((0, 2))


def _compute_radius(points):
    """Compute the packing radius: min(min_pairwise_dist/2, min(1-|ci|))."""
    n = len(points)
    if n < 2:
        return float('inf') if n == 0 else 1.0 - np.linalg.norm(points[0])
    
    tree = cKDTree(points)
    pairs = tree.query_pairs(1.0)  # all pairs within distance 1
    min_pair_dist = float('inf')
    for (i, j) in pairs:
        d = np.linalg.norm(points[i] - points[j])
        min_pair_dist = min(min_pair_dist, d)
    
    dists_center = np.linalg.norm(points, axis=1)
    min_boundary = 1.0 - np.max(dists_center)
    
    return min(min_pair_dist / 2.0, min_boundary)
