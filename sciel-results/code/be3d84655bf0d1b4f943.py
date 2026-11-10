import numpy as np
import time

N = 50
START_TIME = time.time()
TIME_LIMIT = 140.0  # Leave 10s buffer

def _hex_lattice_init(perturb=0.0):
    """Generate a hexagonal lattice configuration."""
    # Estimate spacing based on N
    # Area of unit circle = pi. Area per circle approx pi*r^2.
    # N * pi * r^2 <= pi => r <= 1/sqrt(N) ~ 0.14
    # But packing efficiency is ~0.9, so r ~ 0.13
    # Spacing d = 2r ~ 0.26
    # Let's try a range of spacings
    
    best_points = None
    best_score = -1
    
    # Try different lattice scales
    for scale in np.linspace(0.2, 0.3, 10):
        # Generate points in a grid
        k = int(1.0 / scale) + 2
        rows = []
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * scale
                y = j * scale * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0:
                    rows.append([x, y])
        points = np.array(rows)
        if len(points) < N:
            continue
        
        # Sort by distance from origin to pick the most central ones
        radii = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(radii)
        points = points[indices[:N]]
        
        # Add perturbation
        if perturb > 0:
            points += np.random.randn(N, 2) * perturb
        
        # Calculate score
        score = _calc_score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
            
    return best_points

def _calc_score(points):
    """Calculate the minimum radius for a given configuration."""
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    
    # Boundary distances
    radii = np.hypot(points[:, 0], points[:, 1])
    min_boundary_dist = np.min(1.0 - radii)
    
    return min(min_pair_dist / 2.0, min_boundary_dist)

def _optimize(points, max_steps=10000, initial_step=0.05):
    """Gradient ascent on the minimum radius."""
    step = initial_step
    best_score = _calc_score(points)
    best_points = points.copy()
    no_improve_count = 0
    
    for _ in range(max_steps):
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
        # Compute pairwise distances and boundary distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        radii = np.hypot(points[:, 0], points[:, 1])
        boundary_dist = 1.0 - radii
        
        min_pair_dist = np.min(dist)
        min_boundary_dist = np.min(boundary_dist)
        
        current_score = min(min_pair_dist / 2.0, min_boundary_dist)
        
        if current_score > best_score + 1e-9:
            best_score = current_score
            best_points = points.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 100:
                break
        
        # Determine active constraint
        if min_pair_dist / 2.0 < min_boundary_dist:
            # Inter-circle constraint is active
            idx_i, idx_j = np.unravel_index(np.argmin(dist), dist.shape)
            if idx_i > idx_j:
                idx_i, idx_j = idx_j, idx_i
            
            vec = points[idx_i] - points[idx_j]
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                vec /= norm
            else:
                vec = np.array([1.0, 0.0])
            
            # Move i away from j, j away from i
            points[idx_i] += step * vec
            points[idx_j] -= step * vec
        else:
            # Boundary constraint is active
            idx_i = np.argmin(boundary_dist)
            vec = -points[idx_i]
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                vec /= norm
            else:
                # If at origin, move randomly? Unlikely to be min boundary dist
                vec = np.array([1.0, 0.0])
            
            points[idx_i] += step * vec
        
        # Decay step size
        step *= 0.99
        
    return best_points

def construct():
    best_points = None
    best_score = -1
    
    # Try multiple restarts with different perturbations
    for restart in range(10):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
            
        # Initial lattice with some perturbation
        points = _hex_lattice_init(perturb=0.01 * (restart + 1))
        
        if points is None:
            continue
        
        # Optimize
        optimized_points = _optimize(points)
        score = _calc_score(optimized_points)
        
        if score > best_score:
            best_score = score
            best_points = optimized_points.copy()
            
    return best_points
