"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer for 120s limit
    
    # Initial guess: Grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # Function to compute the minimum radius
    def min_radius(c):
        # Pairwise distances
        # c is (N, 3)
        # d is (N, N)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        # Set diagonal to inf
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)
        
        # Wall distances
        # Distance to boundary [-0.5, 0.5]
        # dist_x = min(x + 0.5, 0.5 - x)
        # min_dist_wall = min over all coords
        dist_to_walls = np.min(np.minimum(c + 0.5, 0.5 - c), axis=1)
        min_wall_dist = np.min(dist_to_walls)
        
        return min(min_pair_dist / 2.0, min_wall_dist)

    current_r = min_radius(centers)
    best_centers = centers.copy()
    best_r = current_r
    
    # Simulated Annealing / Local Search
    T = 0.01  # Initial temperature
    cooling_rate = 0.995
    min_T = 1e-6
    
    while time.time() - start_time < time_limit and T > min_T:
        # Calculate subgradient direction for improvement
        # Find active constraints
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        
        dist_to_walls = np.min(np.minimum(centers + 0.5, 0.5 - centers), axis=1)
        
        r = min(np.min(d) / 2.0, np.min(dist_to_walls))
        
        # Gradient accumulation
        G = np.zeros_like(centers)
        
        # Pair constraints
        # Active if d_ij < 2*r + eps
        eps = 1e-4
        for i in range(N):
            for j in range(i + 1, N):
                if d[i, j] < 2 * r + eps:
                    # Move i away from j, j away from i
                    vec = centers[i] - centers[j]
                    norm = d[i, j]
                    if norm > 1e-9:
                        G[i] += vec / norm
                        G[j] -= vec / norm
        
        # Wall constraints
        for i in range(N):
            if dist_to_walls[i] < r + eps:
                # Move towards center
                # Gradient of min(min(x+0.5, 0.5-x), ...)
                # If x > 0, wall is 0.5-x, grad is -1. Move towards center (negative x).
                # If x < 0, wall is x+0.5, grad is 1. Move towards center (positive x).
                # So grad is -sign(x)
                for k in range(3):
                    if centers[i, k] > 0:
                        G[i, k] -= 1.0
                    else:
                        G[i, k] += 1.0
        
        # Normalize G
        norm_G = np.linalg.norm(G)
        if norm_G > 1e-9:
            G /= norm_G
        
        # Propose move
        # Step size proportional to T?
        step_size = 0.01 * (T / 0.01) + 0.001
        # Add some random noise to escape local minima
        noise = np.random.normal(0, 0.005, centers.shape)
        new_centers = centers + step_size * G + noise
        
        # Clip to cube
        new_centers = np.clip(new_centers, -0.5 + 1e-9, 0.5 - 1e-9)
        
        new_r = min_radius(new_centers)
        
        # Accept/Reject
        delta = new_r - r
        if delta > 0 or np.random.rand() < np.exp(-delta / T):
            centers = new_centers
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centers = centers.copy()
        
        T *= cooling_rate
        
    # Final polish phase: run more iterations with smaller step sizes
    if time.time() - start_time < time_limit - 5.0:
        # Use a smaller step size and run for remaining time
        polish_start = time.time()
        polish_limit = time_limit - 5.0
        step_size_polish = 0.001
        noise_scale_polish = 0.001
        
        while time.time() - polish_start < (polish_limit - polish_start) and time.time() - start_time < time_limit:
            # Calculate subgradient direction
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            d = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(d, np.inf)
            
            dist_to_walls = np.min(np.minimum(centers + 0.5, 0.5 - centers), axis=1)
            r = min(np.min(d) / 2.0, np.min(dist_to_walls))
            
            G = np.zeros_like(centers)
            eps = 1e-5
            for i in range(N):
                for j in range(i + 1, N):
                    if d[i, j] < 2 * r + eps:
                        vec = centers[i] - centers[j]
                        norm = d[i, j]
                        if norm > 1e-9:
                            G[i] += vec / norm
                            G[j] -= vec / norm
            
            for i in range(N):
                if dist_to_walls[i] < r + eps:
                    for k in range(3):
                        if centers[i, k] > 0:
                            G[i, k] -= 1.0
                        else:
                            G[i, k] += 1.0
            
            norm_G = np.linalg.norm(G)
            if norm_G > 1e-9:
                G /= norm_G
            
            noise = np.random.normal(0, noise_scale_polish, centers.shape)
            new_centers = centers + step_size_polish * G + noise
            new_centers = np.clip(new_centers, -0.5 + 1e-9, 0.5 - 1e-9)
            
            new_r = min_radius(new_centers)
            
            if new_r > r:
                centers = new_centers
                current_r = new_r
                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers.copy()
    
    return best_centers
# EVOLVE-BLOCK-END
