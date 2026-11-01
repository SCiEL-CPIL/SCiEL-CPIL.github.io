"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    # 1. Initialize with a hexagonal lattice
    # We want to fill the unit circle.
    # Let's estimate the spacing. Area ~ pi. 50 circles of radius r.
    # r ~ sqrt(pi / 50) / 2 ~ 0.125.
    # Spacing ~ 2r ~ 0.25.
    
    def hex_init():
        # Generate a large hexagonal grid and filter
        spacing = 0.25
        k = int(1.0 / spacing) + 2
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = i * spacing + (j % 2) * spacing * 0.5
                y = j * spacing * np.sqrt(3) / 2
                if np.hypot(x, y) <= 1.0 - spacing/2:
                    points.append([x, y])
        
        points = np.array(points)
        # Sort by distance to center to pick the most central ones first?
        # Or just take the first N.
        # To make it more symmetric, we can try to select points that cover the area evenly.
        # For now, just take the first N from the grid.
        if len(points) < N:
            # If not enough, reduce spacing
            spacing *= 0.95
            return hex_init()
            
        # We want exactly N.
        # A simple selection might be biased. 
        # Let's pick the N points that are closest to the origin? 
        # This tends to create a solid blob. 
        # Better: pick points that maximize the minimum distance among themselves? That's the hard problem.
        # Let's just take the first N from a spiral order or similar.
        # For simplicity, take the N closest to the center. This ensures they are inside.
        dists = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists)[:N]
        return points[indices]

    points = hex_init()
    
    def get_radius(pts):
        d_boundary = 1.0 - np.linalg.norm(pts, axis=1)
        r_boundary = np.min(d_boundary)
        
        # Pairwise distances
        dists = np.sqrt(np.sum((pts[:, np.newaxis, :] - pts[np.newaxis, :, :])**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        r_pair = min_dist / 2.0
        
        return min(r_boundary, r_pair)

    best_points = points.copy()
    best_r = get_radius(best_points)
    
    start_time = time.time()
    step_size = 0.05
    min_step_size = 1e-6
    max_time = 115.0 # Leave 5s buffer
    
    # Hill Climbing Loop
    while time.time() - start_time < max_time:
        # Calculate forces
        forces = np.zeros_like(points)
        
        # Current minimum radius
        current_r = get_radius(points)
        
        # If current is better than best, update best
        if current_r > best_r:
            best_r = current_r
            best_points = points.copy()
            
        # Identify bottleneck pairs and boundary points
        dists = np.sqrt(np.sum((points[:, np.newaxis, :] - points[np.newaxis, :, :])**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Find min distance
        min_d = np.min(dists)
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        
        # Boundary check
        norms = np.linalg.norm(points, axis=1)
        max_norm = np.max(norms)
        max_norm_idx = np.argmax(norms)
        
        # Target radius for force calculation
        # We want to push apart the closest pair and pull in the outermost point
        # But we also need to maintain global structure.
        
        # Force from closest pair
        i, j = min_idx
        if i != j:
            d = dists[i, j]
            if d > 1e-9:
                vec = points[j] - points[i]
                unit_vec = vec / d
                # Strong repulsion
                # Scale by how close they are to the ideal 2*best_r
                # If d < 2*best_r, push hard.
                f_mag = 10.0 / (d * d)
                forces[i] -= unit_vec * f_mag
                forces[j] += unit_vec * f_mag
        
        # Force from boundary
        if max_norm > 1.0 - best_r:
            d_center = norms[max_norm_idx]
            if d_center > 1e-9:
                unit_center = points[max_norm_idx] / d_center
                # Strong inward force
                f_in = 10.0 * (max_norm - (1.0 - best_r))
                forces[max_norm_idx] -= unit_center * f_in
        
        # Weak global repulsion to prevent clustering
        # Only apply to pairs closer than some threshold to save computation? 
        # N=50 is small, so we can do all.
        # But we already did the min pair. 
        # Let's add a weak repulsion for all pairs to maintain even spacing.
        for i_idx in range(N):
            for j_idx in range(i_idx + 1, N):
                d = dists[i_idx, j_idx]
                if d < 1e-9 or (i_idx, j_idx) == (i, j) or (i_idx, j_idx) == (j, i):
                    continue
                # Weak repulsion
                vec = points[j_idx] - points[i_idx]
                unit_vec = vec / d
                f_mag = 0.1 / (d * d)
                forces[i_idx] -= unit_vec * f_mag
                forces[j_idx] += unit_vec * f_mag

        # Normalize forces to have a reasonable magnitude
        force_mags = np.linalg.norm(forces, axis=1)
        max_force = np.max(force_mags)
        if max_force > 1e-9:
            forces = forces / max_force * 0.5 # Scale to max step of 0.5 initially? No, step_size handles this.
        
        # Propose new points
        new_points = points + forces * step_size
        
        # Ensure validity (inside unit circle)
        new_norms = np.linalg.norm(new_points, axis=1)
        for i_idx in range(N):
            if new_norms[i_idx] > 1.0:
                new_points[i_idx] = new_points[i_idx] / new_norms[i_idx] * 0.9999

        new_r = get_radius(new_points)
        
        if new_r > current_r:
            # Accept move
            points = new_points
            # Maybe increase step size slightly if we are improving fast?
            # step_size = min(0.1, step_size * 1.05)
        else:
            # Reject move, revert
            # Decrease step size
            step_size *= 0.9
            if step_size < min_step_size:
                # If step size is too small, we are stuck.
                # Try a random perturbation?
                # Or break?
                # Let's try a small random kick to escape local minima
                points = points + np.random.normal(0, 0.01, points.shape)
                # Re-clamp
                norms = np.linalg.norm(points, axis=1)
                for i_idx in range(N):
                    if norms[i_idx] > 1.0:
                        points[i_idx] = points[i_idx] / norms[i_idx] * 0.9999
                step_size = 0.01 # Reset step size for the kick

    return best_points
# EVOLVE-BLOCK-END
