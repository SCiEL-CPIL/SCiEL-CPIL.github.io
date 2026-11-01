"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 150.0  # seconds
    safety_margin = 5.0  # seconds before deadline
    
    # Start with a good initial configuration: FCC lattice based
    # The best known packing has radius ~0.11138
    # Let's use a layered approach inspired by known packings
    
    # Generate initial points using a combination of FCC and BCC lattices
    # then refine with gradient-based optimization
    
    np.random.seed(42)
    
    # Create a dense grid of candidate positions
    # Use a face-centered cubic (FCC) like arrangement
    
    # Start with a simple cubic grid and perturb
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Take first N points
    centers = grid[:N].copy()
    
    # Add small random perturbation
    centers += np.random.randn(N, 3) * 0.01
    
    # Clip to stay within bounds
    centers = np.clip(centers, -0.45, 0.45)
    
    best_centers = centers.copy()
    best_radius = _compute_radius(centers)
    
    # Iterative optimization: push spheres apart and keep them in bounds
    max_iterations = 5000
    lr = 0.001
    
    for it in range(max_iterations):
        if time.time() > deadline - safety_margin:
            break
        
        # Compute pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Find closest pairs
        np.fill_diagonal(dists, np.inf)
        
        # Compute forces: repulsion for close pairs, attraction to center for boundary
        forces = np.zeros_like(centers)
        
        # Repulsion between close spheres
        for i in range(N):
            for j in range(i + 1, N):
                d = dists[i, j]
                if d < 0.05:  # Only consider close pairs for efficiency
                    # Force magnitude proportional to overlap
                    overlap = 0.05 - d
                    if d > 1e-10:
                        force_dir = (centers[i] - centers[j]) / d
                        forces[i] += force_dir * overlap
                        forces[j] -= force_dir * overlap
        
        # Boundary forces: push back if too close to walls
        for axis in range(3):
            too_close_left = centers[:, axis] < -0.45
            too_close_right = centers[:, axis] > 0.45
            forces[too_close_left, axis] += 0.01
            forces[too_close_right, axis] -= 0.01
        
        # Update positions
        centers += lr * forces
        
        # Keep within bounds
        centers = np.clip(centers, -0.49, 0.49)
        
        # Compute current radius
        current_radius = _compute_radius(centers)
        
        if current_radius > best_radius:
            best_radius = current_radius
            best_centers = centers.copy()
    
    # Final polish: local optimization
    best_centers = _polish(best_centers, deadline - safety_margin)
    final_radius = _compute_radius(best_centers)
    
    if final_radius > best_radius:
        best_centers = best_centers
        best_radius = final_radius
    
    return best_centers


def _compute_radius(centers):
    """Compute the maximum radius for given centers."""
    N = len(centers)
    if N == 0:
        return 0.0
    
    # Minimum distance between any pair of centers
    min_dist = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_dist:
                min_dist = d
    
    # Minimum distance to boundary
    min_boundary = np.inf
    for c in centers:
        for k in range(3):
            d = 0.5 - abs(c[k])
            if d < min_boundary:
                min_boundary = d
    
    return min(min_dist / 2.0, min_boundary)


def _polish(centers, time_budget):
    """Local optimization to improve the packing."""
    start_time = time.time()
    N = len(centers)
    centers = centers.copy()
    
    # Use a simple iterative method: for each sphere, try to move it to increase minimum distance
    lr = 0.005
    max_iter = 2000
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget - 2.0:
            break
        
        # Compute all pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Find the closest pair
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_idx[0], min_idx[1]
        if i > j:
            i, j = j, i
        
        d = dists[i, j]
        
        # Also check boundary constraints
        boundary_violation = False
        for k in range(3):
            if 0.5 - abs(centers[i, k]) < d / 2:
                boundary_violation = True
            if 0.5 - abs(centers[j, k]) < d / 2:
                boundary_violation = True
        
        if d < 0.1 or boundary_violation:
            # Move i and j apart
            if d > 1e-10:
                force_dir = (centers[i] - centers[j]) / d
                centers[i] += force_dir * lr * 0.5
                centers[j] -= force_dir * lr * 0.5
            else:
                centers[i] += np.random.randn(3) * 0.01
                centers[j] -= np.random.randn(3) * 0.01
        
        # Keep within bounds
        centers = np.clip(centers, -0.499, 0.499)
    
    return centers

# EVOLVE-BLOCK-END
