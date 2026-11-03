import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    best_radius = 0.0
    best_centers = None
    
    # Initial configurations
    # 1. Grid based
    # 2. Random
    
    def compute_radius(centers):
        # Centers are (N, 3)
        # Compute pairwise distances
        # Use broadcasting for efficiency
        # dist[i, j] = ||c_i - c_j||
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Mask diagonal
        np.fill_diagonal(dists, np.inf)
        
        # Min pairwise distance / 2
        min_pair_dist = np.min(dists) / 2.0
        
        # Min distance to walls
        # Walls at +/- 0.5
        # dist_wall_i = min(0.5 - |c_ik|) for k in 0,1,2
        # centers are in [-0.5, 0.5]
        # 0.5 - |c| is distance to nearest wall in that dimension
        dist_to_walls = 0.5 - np.abs(centers)
        min_wall_dist = np.min(dist_to_walls)
        
        return min(min_pair_dist, min_wall_dist)

    def local_search(centers, steps=2000, lr_start=0.05, lr_end=0.001):
        centers = centers.copy()
        centers = np.clip(centers, -0.5, 0.5)
        
        for step in range(steps):
            # Check time
            if time.time() > deadline - 5.0:
                break
            
            # Compute forces
            # Pairwise repulsion: F_ij = (c_i - c_j) / ||c_i - c_j||^2
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists_sq = np.sum(diff**2, axis=2)
            np.fill_diagonal(dists_sq, 1.0) # Avoid div by 0, we will mask later
            
            # Force vector F_i = sum_j (c_i - c_j) / dist_ij^2
            # We can compute this as: sum_j diff[i,j] / dists_sq[i,j]
            # But diff is (N, N, 3).
            # F = sum_j (diff[i,j] / dists_sq[i,j])
            # To avoid division by zero issues, we can use a softening or just ensure dist > 0.
            # Since we clip, dist can be 0 if centers coincide.
            # Let's add a small epsilon to denominator.
            forces = np.zeros_like(centers)
            
            # Vectorized force calculation
            # dists_sq is (N, N)
            # We want to sum over j for each i.
            # F_i = sum_j (c_i - c_j) / (||c_i - c_j||^2 + eps)
            # This is equivalent to:
            # F_i = N*c_i - sum_j c_j / (dist^2 + eps) ... no, that's not right.
            # (c_i - c_j) / d^2 = c_i/d^2 - c_j/d^2
            # Sum_j c_i/d^2 = c_i * sum_j (1/d^2)
            # Sum_j c_j/d^2 = sum_j (c_j * (1/d^2))
            
            inv_dists_sq = 1.0 / (dists_sq + 1e-6)
            np.fill_diagonal(inv_dists_sq, 0.0)
            
            # Sum of 1/d^2 for each i
            sum_inv_dists_sq = np.sum(inv_dists_sq, axis=1) # (N,)
            
            # Sum of c_j / d^2 for each i
            # c_j is (N, 3). We need sum_j (c_j * inv_dists_sq[i,j])
            # This is matrix multiplication: inv_dists_sq (N, N) @ c_j (N, 3)
            sum_c_j_over_d_sq = inv_dists_sq @ centers # (N, 3)
            
            # Force from spheres
            F_spheres = centers * sum_inv_dists_sq[:, np.newaxis] - sum_c_j_over_d_sq
            
            # Force from walls
            # For each coordinate k, if c_ik > 0, force is -1/(0.5 - c_ik)^2
            # if c_ik < 0, force is +1/(0.5 + c_ik)^2
            # Let d_wall = 0.5 - |c_ik|
            # Force direction is towards center: -sign(c_ik)
            # Magnitude is 1/d_wall^2
            # F_wall_ik = -sign(c_ik) / (0.5 - |c_ik|)^2
            
            # Avoid division by zero if center is exactly on wall
            abs_c = np.abs(centers)
            d_wall = 0.5 - abs_c
            d_wall = np.maximum(d_wall, 1e-6)
            
            F_walls = -np.sign(centers) / (d_wall**2)
            
            # Total force
            F_total = F_spheres + F_walls
            
            # Learning rate decay
            lr = lr_start * (lr_end / lr_start) ** (step / steps)
            
            # Update
            centers += lr * F_total
            
            # Project back to cube
            centers = np.clip(centers, -0.5, 0.5)
            
        return centers

    # Generate initial candidates
    candidates = []
    
    # 1. Grid 5x5x4
    m1, m2, m3 = 5, 5, 4
    ticks1 = (np.arange(m1) + 0.5) / m1 - 0.5
    ticks2 = (np.arange(m2) + 0.5) / m2 - 0.5
    ticks3 = (np.arange(m3) + 0.5) / m3 - 0.5
    grid = np.array([(x, y, z) for z in ticks3 for y in ticks2 for x in ticks1])
    candidates.append(grid[:N])
    
    # 2. Grid 4x5x5
    m1, m2, m3 = 4, 5, 5
    ticks1 = (np.arange(m1) + 0.5) / m1 - 0.5
    ticks2 = (np.arange(m2) + 0.5) / m2 - 0.5
    ticks3 = (np.arange(m3) + 0.5) / m3 - 0.5
    grid = np.array([(x, y, z) for z in ticks3 for y in ticks2 for x in ticks1])
    candidates.append(grid[:N])
    
    # 3. Random
    np.random.seed(42)
    candidates.append(np.random.uniform(-0.45, 0.45, (N, 3)))
    
    # 4. Jittered Grid
    base_grid = candidates[0].copy()
    np.random.seed(123)
    jitter = np.random.normal(0, 0.01, (N, 3))
    candidates.append(np.clip(base_grid + jitter, -0.5, 0.5))
    
    for i, init_centers in enumerate(candidates):
        if time.time() > deadline - 10.0:
            break
            
        # Run local search
        opt_centers = local_search(init_centers, steps=3000)
        
        # Calculate radius
        r = compute_radius(opt_centers)
        
        if r > best_radius:
            best_radius = r
            best_centers = opt_centers.copy()
            
        # Try a second pass with smaller steps if time permits
        if time.time() < deadline - 20.0:
            opt_centers2 = local_search(opt_centers, steps=2000, lr_start=0.01, lr_end=0.0001)
            r2 = compute_radius(opt_centers2)
            if r2 > best_radius:
                best_radius = r2
                best_centers = opt_centers2.copy()

    # Final check
    if best_centers is None:
        # Fallback to simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]

    return best_centers
