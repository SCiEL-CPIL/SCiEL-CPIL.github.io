"""Program for packing 200 equal circles in a unit square using Deterministic Force Relaxation."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    # Initialize with a jittered grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    rng = np.random.default_rng(42)
    centers += rng.normal(0, 0.005, size=centers.shape)
    centers = np.clip(centers, -0.499, 0.499)

    current_centers = centers.copy()
    
    # Parameters
    max_iter = 2000
    step_size = 0.01
    decay = 0.999
    
    for i in range(max_iter):
        # Compute pairwise distances
        diff = current_centers[:, np.newaxis, :] - current_centers[np.newaxis, :]
        dists_sq = np.sum(diff ** 2, axis=-1)
        np.fill_diagonal(dists_sq, 1.0) # Avoid division by zero
        
        # Repulsion forces: F_ij proportional to (c_i - c_j) / d_ij^3
        # Using d^2 in denominator for stability: (c_i - c_j) / d_ij^4 * d_ij? 
        # Let's use standard Coulomb: 1/d^2 force magnitude, direction (c_i-c_j)/d.
        # Force vector = (c_i - c_j) / d^3.
        # We have diff = c_i - c_j. dists_sq = d^2.
        # Force = diff / (dists_sq * sqrt(dists_sq)) = diff / dists_sq^1.5
        
        # Vectorized force calculation
        # diff shape (N, N, 2)
        # dists_sq shape (N, N)
        # inv_dists_cubed = 1 / dists_sq^1.5
        inv_dists_cubed = 1.0 / (dists_sq ** 1.5)
        # Set diagonal to 0
        inv_dists_cubed[np.diag_indices_from(inv_dists_cubed)] = 0.0
        
        # Forces from other particles
        # force_i = sum_j (c_i - c_j) / d_ij^3
        # diff is (c_i - c_j). So we sum diff * inv_dists_cubed over j.
        forces = np.sum(diff * inv_dists_cubed[:, :, np.newaxis], axis=1)
        
        # Wall repulsion
        # V_wall = 1/(0.5-x)^2 + 1/(0.5+x)^2
        # dV/dx = 2/(0.5-x)^3 - 2/(0.5+x)^3
        x = current_centers[:, 0]
        y = current_centers[:, 1]
        
        # Safe denominators
        dx_pos = np.maximum(0.5 - x, 1e-6)
        dx_neg = np.maximum(0.5 + x, 1e-6)
        dy_pos = np.maximum(0.5 - y, 1e-6)
        dy_neg = np.maximum(0.5 + y, 1e-6)
        
        # Gradient of wall energy (points inward)
        # Force should push away from wall.
        # If x > 0, wall is at 0.5. Force is negative (left).
        # dV/dx = 2/(0.5-x)^3 - 2/(0.5+x)^3.
        # If x is close to 0.5, (0.5-x) is small, term is large positive.
        # Wait, potential V = 1/(0.5-x)^2. Gradient dV/dx = 2/(0.5-x)^3.
        # Force F = -dV/dx. So F_x = -2/(0.5-x)^3 + 2/(0.5+x)^3.
        
        force_x = -2.0 / (dx_pos ** 3) + 2.0 / (dx_neg ** 3)
        force_y = -2.0 / (dy_pos ** 3) + 2.0 / (dy_neg ** 3)
        
        forces[:, 0] += force_x
        forces[:, 1] += force_y
        
        # Update positions
        current_centers += step_size * forces
        
        # Clip to square
        current_centers = np.clip(current_centers, -0.5, 0.5)
        
        # Decay step size
        step_size *= decay
        if step_size < 1e-6:
            break

    return current_centers
# EVOLVE-BLOCK-END
