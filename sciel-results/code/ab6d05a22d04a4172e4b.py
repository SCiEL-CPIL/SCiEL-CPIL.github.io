"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def _init_hex_lattice():
    s = np.sqrt(2 * np.pi / (N * np.sqrt(3)))
    points = []
    k = int(1.0 / s) + 2
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 1.0:
                points.append([x, y])
    points = np.array(points)
    if len(points) > N:
        radii = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(radii)
        points = points[indices[:N]]
    elif len(points) < N:
        # Fallback: reduce spacing
        while len(points) < N:
            s *= 0.95
            points = []
            k = int(1.0 / s) + 2
            for i in range(-k, k + 1):
                for j in range(-k, k + 1):
                    x = (i + 0.5 * (j % 2)) * s
                    y = j * s * np.sqrt(3.0) / 2.0
                    if x*x + y*y <= 1.0:
                        points.append([x, y])
            points = np.array(points)
            if len(points) > N:
                radii = np.hypot(points[:, 0], points[:, 1])
                indices = np.argsort(radii)
                points = points[indices[:N]]
    return points

def _compute_radius(points):
    if len(points) < 2:
        return 1.0 - np.max(np.hypot(points[:, 0], points[:, 1]))
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    radii = np.hypot(points[:, 0], points[:, 1])
    min_bound = 1.0 - np.max(radii)
    
    return min(min_dist / 2.0, min_bound)

def _optimize(points, r_constraint, max_iters=200):
    points = points.copy()
    R_bound = 1.0 - r_constraint
    N = len(points)
    step_size = 0.01
    
    for _ in range(max_iters):
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        min_dists = np.min(dists, axis=1)
        closest_indices = np.argmin(dists, axis=1)
        
        closest_points = points[closest_indices]
        vecs = points - closest_points
        norms = np.linalg.norm(vecs, axis=1)
        norms[norms < 1e-9] = 1e-9
        forces = vecs / norms[:, np.newaxis]
        
        points += forces * step_size
        
        radii = np.linalg.norm(points, axis=1)
        mask = radii > R_bound
        if np.any(mask):
            points[mask] = points[mask] * (R_bound / radii[mask])[:, np.newaxis]
            
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    points = _init_hex_lattice()
    current_r = _compute_radius(points)
    best_r = current_r
    best_points = points.copy()
    
    stagnation = 0
    
    while time.time() < deadline:
        # Try to optimize with current best_r
        # Use a slightly smaller constraint to allow movement?
        # No, use best_r.
        
        new_points = _optimize(best_points, best_r, max_iters=100)
        new_r = _compute_radius(new_points)
        
        if new_r > best_r + 1e-9:
            best_r = new_r
            best_points = new_points
            stagnation = 0
        else:
            stagnation += 1
            if stagnation > 10:
                # Perturb
                noise = np.random.normal(0, 0.05 * best_r, best_points.shape)
                perturbed = best_points + noise
                # Project to unit circle
                radii = np.linalg.norm(perturbed, axis=1)
                mask = radii > 1.0
                if np.any(mask):
                    perturbed[mask] = perturbed[mask] * (1.0 / radii[mask])[:, np.newaxis]
                best_points = perturbed
                stagnation = 0
                
    return best_points

# EVOLVE-BLOCK-END
