"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for safety
    
    # 1. Initialization: Use a grid-like start to avoid bad local minima
    # We want 50 points. A 4x4x4 grid has 64 points. We can take a subset or use a denser grid.
    # Let's use a 4x4x4 grid and randomly select 50, or just use the first 50 in a specific order.
    # Better: Generate a 4x4x4 grid, shuffle, and take 50.
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Shuffle to break symmetry
    np.random.seed(42)
    np.random.shuffle(grid)
    centers = grid[:N].copy()
    
    # Ensure centers are strictly inside
    centers = np.clip(centers, -0.49, 0.49)
    
    def score(centers):
        # Calculate min pairwise distance
        # dists[i, j] = ||c_i - c_j||
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        # Calculate min wall distance
        # Walls are at +/- 0.5
        # dist to wall for axis k is 0.5 - |c_ik|
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        
        return min(min_pair, min_wall)
    
    best_score = score(centers)
    best_centers = centers.copy()
    
    # Parameters for Langevin optimization
    # We want to maximize r.
    # Gradient ascent on r.
    # r = min(min_pair, min_wall)
    # If min_pair < min_wall, we need to increase min_pair.
    # If min_wall < min_pair, we need to increase min_wall.
    
    # Let's use a simple force-based approach:
    # For each pair, if dist < target_dist, push apart.
    # For each wall, if dist < target_dist, push in.
    # But target_dist is the current best score? No, that's static.
    # Better: Use the gradient of the "soft" minimum or just push all "tight" constraints.
    
    # Let's define a function to compute forces.
    # We will use a "temperature" T that decays.
    T = 0.01
    step_size = 0.005
    
    # Precompute indices for pairs to avoid creating full matrix every step?
    # N=50, so 1225 pairs. Full matrix is fine.
    
    current_score = best_score
    
    while time.time() < deadline:
        # Compute forces
        forces = np.zeros_like(centers)
        
        # 1. Pairwise repulsion
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, 1e9) # Avoid div by zero
        
        # We want to increase the minimum distance.
        # So we only care about pairs that are close to the current minimum.
        # Let's define a "soft" minimum weight: w_ij = exp(-(d_ij - current_score)^2 / sigma^2)
        # Or simpler: if d_ij < current_score * 1.1, apply force.
        # Actually, standard gradient ascent on min distance:
        # The gradient of min(d_ij) is the gradient of the argmin term.
        # To be robust, we sum gradients of all terms that are "active" (close to min).
        
        active_mask = dists < (current_score + 0.01) # Heuristic threshold
        
        # Force from pair (i,j) on i is (c_i - c_j) / d_ij
        # Vectorized:
        # diff / dists[..., np.newaxis]
        # But we need to mask.
        
        # To avoid division by zero issues and handle masks, we can do:
        inv_dists = np.zeros_like(dists)
        valid = dists > 1e-9
        inv_dists[valid] = 1.0 / dists[valid]
        
        # Weight by activity
        weights = np.zeros_like(dists)
        weights[active_mask & valid] = 1.0
        
        # Force on i from j is weights[i,j] * (c_i - c_j) / d_ij
        # Sum over j
        # force_i = sum_j w_ij * (c_i - c_j) / d_ij
        # This can be computed as:
        # term = weights * inv_dists
        # force_i = sum_j term[i,j] * (c_i - c_j)
        # c_i is repeated. 
        # Let's use matrix multiplication.
        # (c_i - c_j) = c_i * 1^T - 1 * c_j^T
        # Sum_j term[i,j] * c_i - Sum_j term[i,j] * c_j
        # Sum_j term[i,j] is row sum of term matrix.
        row_sums = np.sum(term, axis=1)
        # Sum_j term[i,j] * c_j is term @ centers
        
        pair_forces = row_sums[:, np.newaxis] * centers - term @ centers
        
        # 2. Wall repulsion
        # dist to wall k for axis a is 0.5 - |c_ik|
        # If dist < current_score + threshold, push in.
        # Wall distances:
        # x-walls: 0.5 - |x|, force direction: sign(x) * -1 (towards 0)
        # y-walls: 0.5 - |y|
        # z-walls: 0.5 - |z|
        
        wall_dists = 0.5 - np.abs(centers)
        # Shape (N, 3)
        
        wall_active = wall_dists < (current_score + 0.01)
        
        # Force direction is -sign(center)
        wall_forces = -np.sign(centers)
        
        # Apply only where active
        wall_forces[~wall_active] = 0.0
        
        # Combine forces
        total_forces = pair_forces + wall_forces
        
        # Normalize forces to prevent instability
        norms = np.linalg.norm(total_forces, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized_forces = total_forces / norms
        
        # Langevin step
        noise = np.random.normal(0, T, size=centers.shape)
        
        new_centers = centers + step_size * normalized_forces + noise
        
        # Project back to cube
        new_centers = np.clip(new_centers, -0.49, 0.49)
        
        # Calculate new score
        new_score = score(new_centers)
        
        if new_score > best_score:
            best_score = new_score
            best_centers = new_centers.copy()
            # If we improve, we can increase step size slightly or keep it
            # If we don't improve, we should decay temperature
            T *= 0.995
            # Step size can also decay or adapt
            # Let's keep step size constant for now, or decay it slightly
            step_size *= 0.999
        else:
            # Even if score drops, we might want to accept it if T is high (Metropolis)
            # But for simplicity and stability, let's just update current state if it's not too bad?
            # Actually, standard gradient ascent rejects worse steps.
            # To escape local minima, the noise is key.
            # Let's always move to new_centers to explore, but track best separately.
            centers = new_centers
            T *= 0.995
            step_size *= 0.999
            
        # Update current score for next iteration's force calculation
        # Using the actual new_score for force calculation is more responsive
        current_score = new_score
        
    return best_centers
# EVOLVE-BLOCK-END
