"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    """
    Find 50 circle centers in [-0.5, 0.5]^2 that maximize the minimum radius.
    
    Strategy:
    1. Generate multiple diverse initial configurations (grid, hexagonal, random).
    2. Use a repulsion-based local search (force-directed) to push circles apart.
    3. Use L-BFGS-B optimization with a smooth approximation of the min-function.
    4. Keep track of the best configuration found within the time budget.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    def compute_radius(centers):
        """Compute the maximum common radius for a given set of centers."""
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pairwise = np.min(dists)
        
        # Boundary clearance
        min_boundary = 0.5 - np.max(np.abs(centers), axis=1).max()
        
        return min(min_pairwise / 2, min_boundary)
    
    def smooth_objective(centers_flat, beta=50):
        """
        Smooth approximation of the negative minimum radius.
        Uses a soft-min to make the objective differentiable.
        """
        c = centers_flat.reshape(N, 2)
        
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
        np.fill_diagonal(dists, np.inf)
        
        # Soft min of pairwise distances / 2
        # Using log-sum-exp trick for stability
        # We want to approximate min(dists) / 2
        # Let's use a simple approach: penalize small distances heavily
        # Actually, let's use the fact that min(x) = -max(-x)
        # And max can be approximated by log-sum-exp
        # But for simplicity, let's use a differentiable penalty
        
        # Boundary constraints
        abs_c = np.abs(c)
        max_abs = np.max(abs_c, axis=1)
        boundary_violation = np.maximum(0, max_abs - 0.5)
        
        # Pairwise distances penalty
        # We want to maximize the minimum distance
        # Let's use a smooth approximation: 
        # f(d) = -d + (1/beta) * log(1 + exp(beta * d)) approximates max(0, d)
        # But we want to maximize min(d), which is equivalent to minimizing -min(d)
        # Let's use: for each pair, penalty = max(0, r_target - d)
        # But r_target is unknown. 
        
        # Alternative: Use the fact that we want to maximize r such that all d_ij >= 2r
        # and boundary >= r.
        # Let's define the objective as the negative of the "bottleneck" radius.
        
        # Simpler approach: just use the raw min but with a smooth version
        # min(dists) can be approximated by:
        # -1/beta * log(sum(exp(-beta * dists)))
        # This is a soft-min.
        
        # Mask out diagonal
        mask = ~np.eye(N, dtype=bool)
        dists_masked = dists[mask]
        
        # Soft min
        min_dists_soft = -1.0 / beta * np.log(np.sum(np.exp(-beta * dists_masked)))
        
        # Boundary soft min
        boundary_clearances = 0.5 - max_abs
        min_boundary_soft = -1.0 / beta * np.log(np.sum(np.exp(-beta * boundary_clearances)))
        
        # Overall soft min
        r_soft = min(min_dists_soft / 2, min_boundary_soft)
        
        # Add heavy penalty for boundary violations
        penalty = np.sum(boundary_violation**2) * 1000
        
        return -r_soft + penalty
    
    def repulsion_search(centers, max_iters=5000, step_size=0.005):
        """
        Force-directed repulsion search.
        Pushes overlapping or close circles apart.
        """
        c = centers.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Compute forces
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            # Force between each pair: repulsive if too close
            # F = k * (d_min - d) * direction, if d < d_min
            # Here d_min is roughly 2 * current_r
            current_r = compute_radius(c)
            d_min = 2 * current_r + 0.01  # Slightly more than touching
            
            forces = np.zeros_like(c)
            for i in range(N):
                for j in range(i + 1, N):
                    d = dists[i, j]
                    if d < d_min and d > 1e-10:
                        direction = (c[j] - c[i]) / d
                        force_mag = step_size * (d_min - d) / d_min
                        forces[i] -= direction * force_mag
                        forces[j] += direction * force_mag
            
            # Boundary forces: push back if too close to boundary
            for i in range(N):
                for k in range(2):
                    if c[i, k] > 0.5 - current_r:
                        forces[i, k] -= step_size * (c[i, k] - (0.5 - current_r))
                    elif c[i, k] < -0.5 + current_r:
                        forces[i, k] += step_size * ((-0.5 + current_r) - c[i, k])
            
            # Apply forces
            c_new = c + forces
            c_new = np.clip(c_new, -0.5 + 1e-9, 0.5 - 1e-9)
            
            # Check if improved
            r_old = current_r
            r_new = compute_radius(c_new)
            
            if r_new > r_old:
                c = c_new
            else:
                # Try smaller step
                c_new = c + forces * 0.5
                c_new = np.clip(c_new, -0.5 + 1e-9, 0.5 - 1e-9)
                r_new = compute_radius(c_new)
                if r_new > r_old:
                    c = c_new
                else:
                    # No improvement, reduce step size
                    step_size *= 0.95
                    if step_size < 1e-6:
                        break
        
        return c
    
    # Initial configurations
    initializers = []
    
    # 1. Grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    initializers.append(grid[:N].copy())
    
    # 2. Grid with jitter
    np.random.seed(42)
    grid_jittered = grid[:N].copy()
    grid_jittered += np.random.normal(0, 0.02, size=grid_jittered.shape)
    grid_jittered = np.clip(grid_jittered, -0.49, 0.49)
    initializers.append(grid_jittered)
    
    # 3. Hexagonal-like
    n_rows = int(np.ceil(np.sqrt(N * 2 / np.sqrt(3))))
    pts = []
    for row in range(n_rows):
        y = -0.5 + (row + 0.5) / n_rows
        offset = (row % 2) * (1.0 / (2 * np.sqrt(3) * n_rows))
        n_cols = int(np.ceil(N / n_rows))
        for col in range(n_cols):
            x = -0.5 + (col + 0.5) / n_cols + offset
            pts.append([x, y])
    hex_grid = np.array(pts[:N])
    hex_grid = np.clip(hex_grid, -0.49, 0.49)
    initializers.append(hex_grid)
    
    # 4. Random
    np.random.seed(123)
    random_init = np.random.uniform(-0.45, 0.45, (N, 2))
    initializers.append(random_init)
    
    # Optimize each initial configuration
    best_centers = None
    best_val = -np.inf
    
    for idx, init in enumerate(initializers):
        if time.time() > deadline - 30:
            break
        
        # First, do repulsion search
        c_rep = repulsion_search(init, max_iters=3000, step_size=0.01)
        r_rep = compute_radius(c_rep)
        
        if r_rep > best_val:
            best_val = r_rep
            best_centers = c_rep.copy()
        
        # Then, do L-BFGS-B optimization starting from the repulsion result
        if time.time() > deadline - 20:
            break
        
        try:
            result = minimize(
                smooth_objective,
                c_rep.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 50000, 'gtol': 1e-5},
                bounds=[(-0.5, 0.5)] * (2 * N)
            )
            
            c_opt = result.x.reshape(N, 2)
            c_opt = np.clip(c_opt, -0.5 + 1e-9, 0.5 - 1e-9)
            r_opt = compute_radius(c_opt)
            
            if r_opt > best_val:
                best_val = r_opt
                best_centers = c_opt.copy()
        
        except Exception as e:
            pass
    
    # Final refinement with more aggressive repulsion
    if best_centers is not None and time.time() < deadline - 15:
        c_final = repulsion_search(best_centers, max_iters=5000, step_size=0.005)
        r_final = compute_radius(c_final)
        
        if r_final > best_val:
            best_val = r_final
            best_centers = c_final.copy()
    
    # Ensure we always return something valid
    if best_centers is None:
        best_centers = initializers[0]
    
    return best_centers
# EVOLVE-BLOCK-END
