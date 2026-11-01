import numpy as np
import time
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # 1. Initial Configuration: 5x5x4 Grid
    # This gives a baseline radius of 0.1
    nx, ny, nz = 5, 5, 4
    # Spacing to fill the cube roughly
    # x: -0.4 to 0.4 (5 points) -> step 0.2
    # y: -0.4 to 0.4 (5 points) -> step 0.2
    # z: -0.3 to 0.3 (4 points) -> step 0.2
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.3, 0.3, nz)
    
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    # Add small random jitter to break symmetry
    np.random.seed(42)
    centers += np.random.normal(0, 0.01, centers.shape)
    centers = np.clip(centers, -0.49, 0.49)
    
    best_centers = centers.copy()
    best_radius = 0.0
    
    def compute_radius(c):
        if not np.all(np.isfinite(c)):
            return 0.0
        if np.any(c > 0.5) or np.any(c < -0.5):
            return 0.0
        # Wall distance
        wall_dists = 0.5 - np.abs(c)
        r_wall = np.min(wall_dists)
        
        # Pairwise distance
        # Using scipy cKDTree for efficiency, though N=100 is small enough for O(N^2)
        # O(N^2) is 10000 ops, very fast. Let's use vectorized numpy for simplicity and speed in tight loops if needed, 
        # but cKDTree is safer if we scale up. For N=100, numpy is fine.
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Mask diagonal
        np.fill_diagonal(dists, np.inf)
        r_pair = np.min(dists) / 2.0
        
        return min(r_wall, r_pair)
    
    current_r = compute_radius(centers)
    if current_r > best_radius:
        best_radius = current_r
        best_centers = centers.copy()
        
    def soft_energy(c, beta):
        """
        Minimize this energy to maximize the minimum distance.
        E = sum(exp(-beta * d_ij)) + sum(exp(-beta * w_i))
        As beta increases, this function becomes dominated by the smallest distances.
        """
        if np.any(c > 0.5) or np.any(c < -0.5):
            return 1e10
        
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Wall distances
        wall_dists = 0.5 - np.abs(c)
        
        # Sum of exponentials
        # Clip distances to avoid overflow in exp if they are very large (though they are bounded by 1.414)
        # Actually, we want to MINIMIZE the sum of exp(-beta * d).
        # Small d -> large exp(-beta * d).
        # So minimizing this sum pushes small d to be larger.
        
        # Use log-sum-exp trick for stability if beta is large, but beta won't be THAT large.
        # Max beta ~ 1000. d ~ 0.1. exp(-100) ~ 3.7e-44. Safe.
        
        pair_term = np.sum(np.exp(-beta * dists))
        wall_term = np.sum(np.exp(-beta * wall_dists))
        
        return pair_term + wall_term

    def gradient_energy(c, beta):
        """
        Analytical gradient of soft_energy.
        E = sum_{i<j} exp(-beta * d_ij) + sum_i exp(-beta * w_i)
        
        d_ij = ||c_i - c_j||
        w_i = min_k (0.5 - |c_ik|)
        
        This gradient is complex due to the min in w_i and the norm in d_ij.
        It's often easier to use numerical gradients or L-BFGS-B which approximates them.
        Given N=100, L-BFGS-B with numerical gradients might be slow if it takes many iterations.
        Let's try providing an approximate gradient or just use L-BFGS-B default.
        
        Actually, for N=100, 300 variables.
        L-BFGS-B with finite differences: 300 evaluations per gradient step.
        Each eval is O(N^2) = 10k ops.
        300 * 10k = 3M ops per gradient.
        If we do 100 iterations, 300M ops.
        This is feasible in Python/NumPy within 150s?
        NumPy is fast. 300M ops is ~0.1-0.5s.
        So we can afford many iterations.
        """
        pass # Not implementing manual gradient, relying on L-BFGS-B

    # Optimization Loop
    # We will run a sequence of optimizations with increasing beta (annealing)
    # This helps escape local minima.
    
    betas = [10, 50, 100, 200, 500, 1000]
    
    for beta in betas:
        if time.time() > deadline - 5:
            break
            
        # Define objective
        def obj(c):
            return soft_energy(c, beta)
            
        # Bounds
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        # Flatten centers for scipy
        c_flat = centers.ravel()
        
        # Run optimization
        try:
            res = minimize(
                obj, 
                c_flat, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
        except Exception as e:
            # If optimization fails, keep previous best
            continue
            
        new_centers = res.x.reshape(N, 3)
        
        # Validate
        if not np.all(np.isfinite(new_centers)):
            continue
        if np.any(new_centers > 0.5) or np.any(new_centers < -0.5):
            continue
            
        new_r = compute_radius(new_centers)
        
        if new_r > best_radius:
            best_radius = new_r
            best_centers = new_centers.copy()
            centers = new_centers.copy() # Continue from this improved state
            
        # Small local refinement step: try to push the most constrained spheres
        # This is optional but might help
        # We'll just rely on the next beta step for further refinement
        
    # Final local search / polishing
    # Use a simple random perturbation + accept if better, with small steps
    # This can fine-tune the radius slightly
    
    current_centers = best_centers.copy()
    current_r = best_radius
    
    # Time remaining for polishing
    time_left = deadline - time.time()
    if time_left > 5:
        step_size = 0.005
        # Number of steps to try
        # Estimate time per step: compute_radius is ~10k ops. Very fast.
        # We can do thousands of steps.
        max_steps = 5000
        for i in range(max_steps):
            if time.time() > deadline - 2:
                break
                
            # Pick a random sphere
            idx = np.random.randint(N)
            
            # Generate a small random move
            move = np.random.normal(0, step_size, 3)
            new_pos = current_centers[idx] + move
            
            # Check bounds
            if new_pos[0] < -0.5 or new_pos[0] > 0.5: continue
            if new_pos[1] < -0.5 or new_pos[1] > 0.5: continue
            if new_pos[2] < -0.5 or new_pos[2] > 0.5: continue
            
            # Apply move
            old_pos = current_centers[idx].copy()
            current_centers[idx] = new_pos
            
            new_r = compute_radius(current_centers)
            
            if new_r > current_r:
                current_r = new_r
                if new_r > best_radius:
                    best_radius = new_r
                    best_centers = current_centers.copy()
            else:
                # Revert
                current_centers[idx] = old_pos
                
            # Occasionally accept worse moves to escape local minima?
            # Simple hill climbing is safer for the final stage.
            # Let's stick to hill climbing for the last few seconds to ensure we don't degrade.

    return best_centers
