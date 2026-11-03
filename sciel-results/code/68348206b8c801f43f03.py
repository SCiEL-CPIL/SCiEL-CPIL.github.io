import numpy as np
import time
import math

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin

    def compute_radius(centers):
        if centers.shape != (N, 2):
            return 0.0
        if not np.all(np.isfinite(centers)):
            return 0.0
        if np.any(np.abs(centers) > 0.5 + 1e-9):
            return 0.0
        
        # Wall distances
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        
        # Pair distances
        # For N=200, full matrix is 40k entries, which is fast enough in NumPy
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        r = min(min_wall, min_pair / 2.0)
        return max(0.0, r)

    def hexagonal_init(n_rows, n_cols, d_scale=1.0):
        d = d_scale / (n_cols + 0.5)
        row_spacing = d * math.sqrt(3) / 2
        total_height = (n_rows - 1) * row_spacing + d
        if total_height > 1.0 + 1e-6:
            return None
        pts = []
        for row in range(n_rows):
            y = -0.5 + d/2 + row * row_spacing
            if row % 2 == 0:
                n_in_row = n_cols
                x_start = -0.5 + d/2
            else:
                n_in_row = n_cols - 1
                x_start = -0.5 + d
            for col in range(n_in_row):
                x = x_start + col * d
                pts.append([x, y])
        pts = np.array(pts)
        if len(pts) >= N:
            pts = pts[:N]
        elif len(pts) < N:
            return None
        return pts

    def square_grid_init(m):
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    def simulate_anneal(initial_points, max_iters=2000, initial_temp=0.005, cooling_rate=0.9995):
        """
        Local neighborhood-based repulsion with simulated annealing.
        Uses a cutoff distance to limit interactions to nearest neighbors,
        reducing the effective dimensionality of the optimization landscape.
        """
        pts = initial_points.copy()
        current_radius = compute_radius(pts)
        best_radius = current_radius
        best_pts = pts.copy()
        
        # Pre-compute indices for potential neighbors? 
        # Since positions change, we must re-evaluate neighbors or use a grid.
        # For N=200, full O(N^2) force calc is ~40k ops/iter.
        # 2000 iters * 40k = 80M ops. NumPy can handle this in < 1s per run.
        # So we can afford full matrix distance checks every step if vectorized.
        
        for i in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # 1. Compute all pair distances
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dists, 1.0) # Avoid div by zero
            
            # 2. Compute forces (Repulsion)
            # We want to push points apart if they are closer than 2*best_radius_estimate
            # Use a soft repulsion: F = k * (target_dist - dist) if dist < target_dist
            # To maximize min dist, we can just push all pairs apart proportional to 1/dist^2
            # But that's slow to converge.
            # Better: Minimize potential energy E = sum(1/d^2) + wall_penalty
            # Gradient descent on Energy.
            
            # Force between i and j: F_ij = k * (pts[i] - pts[j]) / dist^3 (for 1/d^2 potential)
            # Actually, let's use a stiffer potential to avoid overlaps: 
            # F = k * (target_r - dist) * (pts[i] - pts[j]) / dist
            # We don't know target_r exactly, so we use the current min_dist/2 as a proxy?
            # No, just use a constant repulsion strength that decays with distance.
            
            # Vectorized Force Calculation
            # F_i = sum_{j!=i} k * (pts[i] - pts[j]) / dist_ij^3
            # This corresponds to minimizing sum 1/dist_ij^2.
            
            inv_dist3 = np.zeros_like(dists)
            mask = dists < 1.0 # Cutoff to save time? 1.0 is the whole square diagonal approx.
            # For N=200, just use all.
            inv_dist3 = 1.0 / (dists ** 3)
            np.fill_diagonal(inv_dist3, 0.0)
            
            # Force matrix: F[i,j] = inv_dist3[i,j] * (pts[i] - pts[j])
            # F = diff * inv_dist3[:, :, np.newaxis]
            forces = diff * inv_dist3[:, :, np.newaxis]
            
            # Sum forces for each point
            net_forces = np.sum(forces, axis=1)
            
            # Wall forces
            # Push away from walls
            # If x > 0.5 - margin, push left. If x < -0.5 + margin, push right.
            # Soft wall: F_wall = k_wall * (0.5 - |x|) is not right.
            # F_wall should be repulsive when close to wall.
            # Let's add a penalty to the energy: sum(1/(0.5-|x|)^2)
            # Gradient: -2 * sign(x) / (0.5-|x|)^3
            wall_dists = 0.5 - np.abs(pts)
            wall_dists_safe = np.clip(wall_dists, 1e-6, None)
            wall_forces = -2.0 * np.sign(pts) / (wall_dists_safe ** 3)
            
            total_forces = net_forces + wall_forces
            
            # 3. Update positions
            # Metropolis step:
            # Calculate energy change? 
            # Simpler: Gradient descent with momentum + noise.
            # v = momentum * v + learning_rate * force + noise
            # But we don't have velocity state easily in this loop structure without extra vars.
            # Let's use a simple noisy gradient step.
            
            # Determine step size based on current temp
            # We want to accept moves that increase energy (radius decreases) with probability exp(-dE/T)
            # Since we are doing gradient descent on Energy (minimizing E), 
            # a move that decreases E is always accepted.
            # A move that increases E is accepted with prob exp(-dE/T).
            
            # To do this properly, we need to compute E_old and E_new.
            # E_old is not stored. Let's compute it.
            # E = sum(1/d^2) + wall_penalty
            # This is expensive to compute twice.
            
            # Alternative: Just use a fixed learning rate and noise, and keep the best solution found.
            # This is a "hill climbing with restarts" or "random walk with selection".
            
            lr = 0.001
            noise_scale = initial_temp * (cooling_rate ** i)
            
            new_pts = pts + lr * total_forces + noise_scale * np.random.randn(N, 2)
            
            # Enforce bounds (hard constraint)
            new_pts = np.clip(new_pts, -0.5, 0.5)
            
            new_radius = compute_radius(new_pts)
            
            # Accept if better, or with probability based on "temperature" if worse
            # Since we don't have exact energy delta, we approximate:
            # If new_radius > current_radius: Accept
            # Else: Accept with prob exp((new_radius - current_radius) / T)
            # Note: Radius is not Energy. Energy ~ 1/Radius^2 roughly?
            # Let's just use the radius difference directly for a simple heuristic.
            
            if new_radius > current_radius:
                pts = new_pts
                current_radius = new_radius
                if current_radius > best_radius:
                    best_radius = current_radius
                    best_pts = pts.copy()
            else:
                # Probability of accepting a worse move
                # delta_r is negative
                delta_r = new_radius - current_radius
                if noise_scale > 1e-8:
                    prob = math.exp(delta_r / (noise_scale * 10)) # Scaling factor 10 is heuristic
                    if np.random.random() < prob:
                        pts = new_pts
                        current_radius = new_radius
                
        return best_pts, best_radius

    # Main logic
    best_centers = None
    best_radius = 0.0
    
    # Generate initializations
    inits = []
    
    # Hexagonal variations
    for n_rows in range(13, 18):
        n_cols = math.ceil(N / n_rows)
        for d_scale in [0.95, 1.0, 1.05, 1.1]:
            pts = hexagonal_init(n_rows, n_cols, d_scale)
            if pts is not None:
                inits.append(pts)
                
    # Square Grid
    m = int(np.ceil(np.sqrt(N)))
    grid = square_grid_init(m)
    inits.append(grid)
    
    # Random inits
    np.random.seed(42)
    for _ in range(5):
        rand_pts = np.random.uniform(-0.49, 0.49, (N, 2))
        inits.append(rand_pts)
        
    # Optimize
    # We have ~145s. 
    # Each SA run takes ~1-2s (2000 iters).
    # We can run ~50-70 runs.
    # We have ~15 inits. We can run multiple SA passes per init with different temps/lengths.
    
    runs_per_init = 3
    
    for i, init in enumerate(inits):
        if time.time() > deadline - 10:
            break
            
        for run_idx in range(runs_per_init):
            if time.time() > deadline - 5:
                break
            
            # Vary parameters slightly
            max_iters = 1500 + run_idx * 500
            init_temp = 0.005 * (1.0 + 0.2 * run_idx)
            
            centers, r = simulate_anneal(init, max_iters=max_iters, initial_temp=init_temp)
            
            if r > best_radius:
                best_radius = r
                best_centers = centers.copy()
                
    # Final check
    if best_centers is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers

# EVOLVE-BLOCK-END
