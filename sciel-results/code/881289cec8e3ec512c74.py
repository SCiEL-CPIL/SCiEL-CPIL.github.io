"""Rewritten program: soft-constraint gradient descent for circle packing."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np


def _soft_penalty(d, r):
    """Smooth hinge: 0 if d >= r, smooth quadratic-like penalty if d < r."""
    # Use a smooth approximation: max(0, r - d)^2 / (2 * eps) with smooth max
    # Simpler: use (r - d) * (r - d) when r > d, smoothed via softplus
    eps = 1e-4
    diff = r - d
    # softplus approximation of max(0, diff)
    soft = eps * np.log1p(np.exp(diff / eps))
    return soft * soft / (2 * eps)


def construct():
    rng = np.random.default_rng(42)
    
    # Initialize with a perturbed grid to avoid symmetry traps
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # Add small random perturbation to break symmetry
    centers += rng.normal(0, 0.001, size=centers.shape)
    
    # Clip to be safely inside
    centers = np.clip(centers, -0.49, 0.49)
    
    # Target radius to optimize
    # Best known is ~0.0366, let's aim slightly below for stability
    r_target = 0.0365
    
    # Gradient descent with vectorized updates
    n_steps = 5000
    lr = 0.0005
    
    for step in range(n_steps):
        # Compute all pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2) + 1e-12)
        
        # Mask out self-pairs
        mask = ~np.eye(N, dtype=bool)
        
        # Compute penalty for pairs closer than r_target
        # Only consider pairs where dist < r_target
        # For efficiency, we can compute gradient only for violating pairs
        
        # Gradient of penalty w.r.t. center i
        # penalty_ij = soft(r - d_ij)^2 / (2*eps)
        # d_ij = |c_i - c_j|
        # d(penalty)/d(c_i) = 2 * soft(r - d_ij) / eps * (-1) * d(soft)/d(r-d) * d(r-d)/d(c_i)
        # This is getting complex. Let's use a simpler direct approach.
        
        # Alternative: use a simpler repulsive force model
        # Force on i from j: if d_ij < r_target, push apart with force proportional to (r_target - d_ij)
        
        forces = np.zeros_like(centers)
        
        for i in range(N):
            for j in range(i + 1, N):
                d = dist[i, j]
                if d < r_target and d > 1e-8:
                    # Push apart
                    direction = (centers[i] - centers[j]) / d
                    strength = (r_target - d) * 0.5
                    forces[i] += direction * strength
                    forces[j] -= direction * strength
        
        # Wall repulsion: push centers away from walls
        for i in range(N):
            for k in range(2):
                # Left wall: x = -0.5, right wall: x = 0.5
                if centers[i, k] < -0.5 + r_target:
                    forces[i, k] += (r_target - (centers[i, k] + 0.5)) * 0.5
                if centers[i, k] > 0.5 - r_target:
                    forces[i, k] -= (r_target - (0.5 - centers[i, k])) * 0.5
        
        # Update
        centers += lr * forces
        
        # Clip to stay inside
        centers = np.clip(centers, -0.5 + r_target, 0.5 - r_target)
    
    # Final cleanup: ensure no overlaps by local refinement
    # Do a few more iterations with smaller step
    for step in range(2000):
        forces = np.zeros_like(centers)
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt(np.sum((centers[i] - centers[j]) ** 2) + 1e-12)
                if d < r_target and d > 1e-8:
                    direction = (centers[i] - centers[j]) / d
                    strength = (r_target - d) * 0.5
                    forces[i] += direction * strength
                    forces[j] -= direction * strength
        
        for i in range(N):
            for k in range(2):
                if centers[i, k] < -0.5 + r_target:
                    forces[i, k] += (r_target - (centers[i, k] + 0.5)) * 0.5
                if centers[i, k] > 0.5 - r_target:
                    forces[i, k] -= (r_target - (0.5 - centers[i, k])) * 0.5
        
        centers += 0.0001 * forces
        centers = np.clip(centers, -0.5 + r_target, 0.5 - r_target)
    
    # Verify validity
    min_dist = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            min_dist = min(min_dist, d)
    
    min_wall = np.inf
    for i in range(N):
        for k in range(2):
            min_wall = min(min_wall, 0.5 - abs(centers[i, k]))
    
    r = min(min_dist / 2, min_wall)
    
    # If r is less than r_target, we need to adjust
    # Return the centers
    return centers
# EVOLVE-BLOCK-END
