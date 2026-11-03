"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    time_limit = 140.0  # Leave 10s buffer for 150s limit
    
    # Initial guess: BCC lattice
    # BCC lattice has 2 points per unit cell: corners and body center
    # We want to fit N=50 spheres. Let's create a BCC lattice and select the best N points.
    # A cubic grid of side m has m^3 unit cells. Each unit cell contributes 2 points (1 corner + 1 body center).
    # Total points = 2 * m^3. We need to select N points from this.
    # To maximize density, we should pick points that are well-distributed.
    # Let's generate a BCC lattice in a larger box and then scale/clip.
    # Or better: generate a BCC lattice that roughly fits the cube.
    
    # Let's try a different approach: generate a grid of BCC unit cells.
    # For m=3, we have 3x3x3=27 unit cells, 54 points. We need 50.
    # For m=4, we have 4x4x4=64 unit cells, 128 points.
    
    # Let's use m=3, which gives 54 points. We can randomly select 50 or optimize the selection.
    # Actually, let's just generate the BCC lattice and take the first N points, or better,
    # generate a set of candidate points and use a greedy or random selection.
    
    # Simpler: Just use a BCC-like initialization.
    # Let's create a 4x4x4 grid of BCC unit cells, but that's too many.
    # Let's use a 3x3x3 grid of BCC unit cells (54 points) and randomly select 50.
    # Or even better: create a list of all BCC points in a 3x3x3 supercell, and then
    # use a simple random selection or a greedy algorithm to pick 50 that maximize min distance.
    
    # For now, let's just generate the BCC points and take the first N.
    # This might not be optimal, but it's a good start.
    
    m = 3
    # Generate BCC lattice points in a box of size m
    # Corner points: (i, j, k) for i,j,k in [0, m]
    # Body center points: (i+0.5, j+0.5, k+0.5) for i,j,k in [0, m-1]
    
    corner_points = []
    for i in range(m + 1):
        for j in range(m + 1):
            for k in range(m + 1):
                corner_points.append([i, j, k])
    
    body_center_points = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                body_center_points.append([i + 0.5, j + 0.5, k + 0.5])
    
    all_points = np.array(corner_points + body_center_points)
    # Scale to fit in [-0.5, 0.5]
    # The points are in [0, m]. We want to map them to [-0.5, 0.5].
    # So scale by 1/m and shift by -0.5.
    all_points = all_points / m - 0.5
    
    # We have 54 points, we need 50.
    # Let's randomly select 50 points.
    # Or better: use a greedy algorithm to select points that maximize the minimum distance.
    # For simplicity, let's just take the first 50 points.
    # This is not ideal, but it's a start.
    
    # Actually, let's use a simple random selection to avoid bias.
    np.random.seed(42)
    indices = np.random.choice(len(all_points), N, replace=False)
    centers = all_points[indices].copy()
    
    # Function to compute the minimum radius
    def min_radius(c):
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)
        
        # Wall distances
        dist_to_walls = np.min(np.minimum(c + 0.5, 0.5 - c), axis=1)
        min_wall_dist = np.min(dist_to_walls)
        
        return min(min_pair_dist / 2.0, min_wall_dist)

    current_r = min_radius(centers)
    best_centers = centers.copy()
    best_r = current_r
    
    # Phase 1: Simulated Annealing to escape local minima
    T = 0.01
    cooling_rate = 0.995
    min_T = 1e-6
    
    while time.time() - start_time < time_limit * 0.4 and T > min_T:
        # Calculate subgradient direction
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        
        dist_to_walls = np.min(np.minimum(centers + 0.5, 0.5 - centers), axis=1)
        
        r = min(np.min(d) / 2.0, np.min(dist_to_walls))
        
        G = np.zeros_like(centers)
        eps = 1e-4
        
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
        
        step_size = 0.01 * (T / 0.01) + 0.001
        noise = np.random.normal(0, 0.005, centers.shape)
        new_centers = centers + step_size * G + noise
        new_centers = np.clip(new_centers, -0.5 + 1e-9, 0.5 - 1e-9)
        
        new_r = min_radius(new_centers)
        
        delta = new_r - r
        if delta > 0 or np.random.rand() < np.exp(-delta / T):
            centers = new_centers
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centers = centers.copy()
        
        T *= cooling_rate
    
    # Phase 2: Local optimization using scipy
    if time.time() - start_time < time_limit - 20:
        # Define objective function to maximize (so we minimize negative)
        def objective(x):
            c = x.reshape(N, 3)
            r = min_radius(c)
            return -r
        
        # Run local optimization from multiple starting points
        n_restarts = 5
        for restart in range(n_restarts):
            if time.time() - start_time > time_limit - 20:
                break
            
            # Start from current best or a perturbed version
            if restart == 0:
                x0 = best_centers.flatten()
            else:
                x0 = best_centers.flatten() + np.random.normal(0, 0.01, N * 3)
                x0 = np.clip(x0, -0.5 + 1e-9, 0.5 - 1e-9)
            
            # Use L-BFGS-B for constrained optimization
            try:
                result = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-0.5 + 1e-9, 0.5 - 1e-9)] * (N * 3),
                    options={'maxiter': 500, 'ftol': 1e-12}
                )
                
                if result.success or result.fun < objective(x0):
                    new_centers = result.x.reshape(N, 3)
                    new_r = min_radius(new_centers)
                    
                    if new_r > best_r:
                        best_r = new_r
                        best_centers = new_centers.copy()
                        
            except Exception:
                pass
    
    # Phase 3: Final polish with smaller steps
    if time.time() - start_time < time_limit - 5:
        polish_start = time.time()
        polish_limit = time_limit - 5.0
        step_size_polish = 0.0005
        noise_scale_polish = 0.0005
        
        centers = best_centers.copy()
        
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
                if new_r > best_r:
                    best_r = new_r
                    best_centers = centers.copy()
    
    return best_centers
# EVOLVE-BLOCK-END
