"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer for 120s limit
    
    # BCC lattice initialization
    # Generate BCC points in a scaled cube
    # BCC has points at (0,0,0) and (0.5,0.5,0.5) in unit cell
    # Scale to fit [-0.5, 0.5]^3
    
    # Number of cells along each axis to get at least N points
    # BCC has 2 points per cell, so need ceil(N/2) cells total
    # Use a 4x4x4 grid of BCC cells = 4*4*4*2 = 128 points, select 50
    n_cells = 4
    cell_size = 1.0 / n_cells  # Each cell is 1/n_cells
    
    # Generate BCC lattice points
    points = []
    for ix in range(n_cells):
        for iy in range(n_cells):
            for iz in range(n_cells):
                # Corner point
                x0 = -0.5 + (ix + 0.0) * cell_size
                y0 = -0.5 + (iy + 0.0) * cell_size
                z0 = -0.5 + (iz + 0.0) * cell_size
                # Body center
                xc = -0.5 + (ix + 0.5) * cell_size
                yc = -0.5 + (iy + 0.5) * cell_size
                zc = -0.5 + (iz + 0.5) * cell_size
                
                # Only include points strictly inside the cube
                margin = 1e-9
                if (margin < x0 < 0.5 - margin and margin < y0 < 0.5 - margin and margin < z0 < 0.5 - margin):
                    points.append([x0, y0, z0])
                if (margin < xc < 0.5 - margin and margin < yc < 0.5 - margin and margin < zc < 0.5 - margin):
                    points.append([xc, yc, zc])
    
    points = np.array(points)
    
    # If we don't have enough points, fall back to grid
    if len(points) < N:
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        centers = grid[:N].copy()
    else:
        # Select N points with maximum minimum distance (greedy)
        # Start from center of cube
        centers = np.zeros((N, 3))
        centers[0] = [0, 0, 0]
        
        for i in range(1, N):
            # Compute distance from each candidate to all already selected
            dists = np.linalg.norm(points - centers[:i], axis=1)
            min_dists = np.min(dists, axis=1)
            # Pick the point with maximum minimum distance
            best_idx = np.argmax(min_dists)
            centers[i] = points[best_idx]
    
    # Function to compute the minimum radius
    def min_radius(c):
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)
        
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
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        
        dist_to_walls = np.min(np.minimum(centers + 0.5, 0.5 - centers), axis=1)
        
        r = min(np.min(d) / 2.0, np.min(dist_to_walls))
        
        # Gradient accumulation
        G = np.zeros_like(centers)
        
        # Pair constraints
        eps = 1e-4
        for i in range(N):
            for j in range(i + 1, N):
                if d[i, j] < 2 * r + eps:
                    vec = centers[i] - centers[j]
                    norm = d[i, j]
                    if norm > 1e-9:
                        G[i] += vec / norm
                        G[j] -= vec / norm
        
        # Wall constraints
        for i in range(N):
            if dist_to_walls[i] < r + eps:
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
        step_size = 0.01 * (T / 0.01) + 0.001
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
    
    # Final polish phase
    if time.time() - start_time < time_limit - 5.0:
        polish_start = time.time()
        polish_limit = time_limit - 5.0
        step_size_polish = 0.001
        noise_scale_polish = 0.001
        
        while time.time() - polish_start < (polish_limit - polish_start) and time.time() - start_time < time_limit:
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
