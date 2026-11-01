"""Rewritten program: continuous potential energy minimization for sphere packing."""
import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = 150.0  # seconds
    safety_margin = 5.0  # seconds before deadline to stop
    
    # --- Initialization: BCC-like grid for good starting point ---
    # We want 100 spheres. A 5x5x4 grid gives 100 points.
    # Cube is [-0.5, 0.5]^3
    # Let's create a 5x5x4 grid
    nx, ny, nz = 5, 5, 4
    x = np.linspace(-0.5 + 0.1, 0.5 - 0.1, nx)
    y = np.linspace(-0.5 + 0.1, 0.5 - 0.1, ny)
    z = np.linspace(-0.5 + 0.1, 0.5 - 0.1, nz)
    
    # Create meshgrid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    centers = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    centers = centers[:N]
    
    # Add small random perturbation to break symmetry
    np.random.seed(42)
    centers += np.random.randn(N, 3) * 0.01
    
    # --- Potential Energy Minimization ---
    # We want to maximize the minimum distance. 
    # Equivalent to minimizing a potential that penalizes close distances.
    # Use a soft-min approach: minimize sum of (1/d_i)^p for large p, or similar.
    # Better: minimize the sum of squared overlaps with a target radius, or use a repulsive potential.
    
    # Let's use an inverse-power law potential: V = sum_{i<j} (1/d_ij)^alpha
    # But this can be stiff. 
    # Alternative: minimize -log(min distance) approximated by soft-min.
    # Or simply: minimize sum_{i<j} max(0, (2r - d_ij)^2) + wall penalty.
    
    # Let's try a simple gradient descent on a smooth objective.
    # Objective: minimize F = sum_{i<j} (1/d_ij)^2 + sum_i (0.5 - |c_i|)^2 * lambda_wall
    # But we want to maximize r. 
    # Let's use the potential: V = sum_{i<j} (1/d_ij)^alpha
    # Gradient: dV/dc_i = -alpha * sum_{j!=i} (1/d_ij)^(alpha+1) * (c_i - c_j) / d_ij
    #            = -alpha * sum_{j!=i} (c_i - c_j) / d_ij^(alpha+2)
    
    # This potential encourages spreading out.
    # We also need to keep spheres inside the cube.
    # Add a wall potential: V_wall = sum_i sum_k (max(0, |c_ik| - 0.5 + eps))^2 / eps^2
    # Or simpler: project back into the cube after each step.
    
    alpha = 4.0  # Exponent for repulsive potential
    lr = 0.001  # Initial learning rate
    n_iter = 50000
    batch_size = 1  # Full batch
    
    best_radius = 0.0
    best_centers = centers.copy()
    
    # Precompute pair indices for efficiency
    # But for N=100, O(N^2) is fine (10000 pairs)
    
    def compute_radius(centers):
        # Compute min distance between any pair and min distance to wall
        if len(centers) < 2:
            return 0.0
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist)
        
        # Wall distance
        # Distance from center to nearest face
        # Faces are at +/-0.5
        wall_dists = []
        for k in range(3):
            wall_dists.append(0.5 - np.abs(centers[:, k]))
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist / 2.0, min_wall_dist)
    
    # Initial radius
    best_radius = compute_radius(centers)
    
    for it in range(n_iter):
        # Check time
        if time.time() - start_time > deadline - safety_margin:
            break
        
        # Compute gradient of V = sum_{i<j} (1/d_ij)^alpha
        # dV/dc_i = -alpha * sum_{j!=i} (c_i - c_j) / d_ij^(alpha+2)
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, 1.0)  # Avoid division by zero
        
        # Gradient term: (c_i - c_j) / d_ij^(alpha+2)
        # This is a matrix of vectors
        grad_term = diff / dist[:, :, np.newaxis] ** (alpha + 2)
        
        # Sum over j for each i
        grad = -alpha * np.sum(grad_term, axis=1)
        
        # Wall constraint: project back into cube
        # Or add a gradient penalty for walls
        # Let's add a strong repulsive force from walls
        # V_wall = sum_i sum_k (max(0, |c_ik| - 0.5 + margin))^2 / margin^2
        # For simplicity, let's just project
        centers_new = centers - lr * grad
        
        # Project into cube [-0.5, 0.5]^3
        centers_new = np.clip(centers_new, -0.5, 0.5)
        
        # Compute new radius
        new_radius = compute_radius(centers_new)
        
        # Adaptive step size
        if new_radius > best_radius:
            best_radius = new_radius
            best_centers = centers_new.copy()
            centers = centers_new
            # Increase learning rate slightly
            lr = min(lr * 1.01, 0.01)
        else:
            # Decrease learning rate
            lr = lr * 0.95
            # Keep old centers
            centers = centers_new  # Still move, but maybe backtrack?
            # Let's do backtracking: if radius decreased, revert
            # Actually, the potential minimization should generally improve the minimum distance
            # But the radius is min(pair_dist/2, wall_dist)
            # If we clip to walls, wall_dist might be 0.
            # Let's be more careful: don't clip too aggressively
            # Instead, use a soft wall potential in the gradient
        
        # Add wall gradient
        # For each coordinate, if c > 0.5 - eps, push back
        eps = 0.01
        wall_grad = np.zeros_like(centers)
        for k in range(3):
            # Right wall
            mask_r = centers_new[:, k] > 0.5 - eps
            wall_grad[mask_r, k] += (centers_new[mask_r, k] - (0.5 - eps)) / eps
            # Left wall
            mask_l = centers_new[:, k] < -0.5 + eps
            wall_grad[mask_l, k] -= (centers_new[mask_l, k] - (-0.5 + eps)) / eps
        
        # Apply wall gradient
        centers_new = centers - lr * (grad + 100.0 * wall_grad)
        centers_new = np.clip(centers_new, -0.5 + 1e-6, 0.5 - 1e-6)
        
        new_radius = compute_radius(centers_new)
        
        if new_radius > best_radius:
            best_radius = new_radius
            best_centers = centers_new.copy()
            centers = centers_new
        else:
            centers = centers_new
            lr = lr * 0.9
    
    # Final check
    final_radius = compute_radius(best_centers)
    if final_radius > best_radius:
        best_radius = final_radius
    
    # Ensure valid output
    if not np.all(np.isfinite(best_centers)):
        # Fallback to grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
    
    return best_centers
