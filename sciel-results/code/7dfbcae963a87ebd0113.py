import numpy as np
import time

N = 100

def construct():
    # 1. Grid Initialization with Jitter
    # 5x5x4 grid
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.375, 0.375, nz)
    
    centers = np.array([(x, y, z) for x in xs for y in ys for z in zs])
    
    # Add small random jitter to break symmetry
    np.random.seed(42) # Fixed seed for reproducibility in evaluation? No, better to use random for exploration.
    # But for a deterministic "best known" approach, maybe fixed is safer? 
    # The task says "maximize", implying we want the best result. 
    # Randomness can help escape local minima. Let's use a random seed based on time or just default.
    # To be safe and reproducible for debugging, but allow improvement, let's just use np.random without fixed seed.
    centers += np.random.normal(0, 0.001, centers.shape)
    centers = np.clip(centers, -0.5, 0.5)

    r = 0.10
    dr = 0.0001
    max_total_steps = 50000
    step = 0
    converged = False
    
    # Learning rate for gradient descent
    lr = 0.05
    
    # Helper to calculate current radius
    def calc_radius(c):
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        max_abs_coord = np.max(np.abs(c), axis=1)
        min_boundary = 0.5 - np.max(max_abs_coord)
        return min(min_pair / 2.0, min_boundary)

    current_r = calc_radius(centers)
    
    while r < 0.115 and step < max_total_steps:
        # Relaxation loop for current r
        # We want to push spheres apart until all pairs are >= 2r and boundaries are respected.
        # Use a decaying learning rate within the relaxation block?
        
        max_iter_per_r = 200
        lr_curr = 0.05
        
        for it in range(max_iter_per_r):
            # Calculate forces
            forces = np.zeros_like(centers)
            
            # Pairwise repulsion
            for i in range(N):
                for j in range(i + 1, N):
                    d_vec = centers[i] - centers[j]
                    d = np.linalg.norm(d_vec)
                    if d < 2 * r and d > 1e-8:
                        # Repulsive force, stronger if closer
                        # Linear penalty: (2r - d)
                        # Quadratic penalty: (2r - d)^2 might be smoother?
                        # Let's stick to linear for now, it's standard.
                        f_mag = (2 * r - d) / d
                        f_vec = d_vec * f_mag
                        forces[i] += f_vec
                        forces[j] -= f_vec
            
            # Boundary forces
            for i in range(N):
                for k in range(3):
                    c = centers[i, k]
                    # Boundary at 0.5 and -0.5
                    dist_pos = 0.5 - c
                    dist_neg = c - (-0.5)
                    
                    if dist_pos < r:
                        # Push in negative direction (towards center)
                        forces[i, k] -= (r - dist_pos)
                    if dist_neg < r:
                        # Push in positive direction (towards center)
                        forces[i, k] += (r - dist_neg)
            
            # Update centers
            centers += lr_curr * forces
            
            # Clip to boundaries to prevent escaping the cube entirely
            centers = np.clip(centers, -0.5, 0.5)
            
            # Check convergence: if max force is small
            max_force = np.max(np.abs(forces))
            if max_force < 1e-6:
                break
            
            # Decay learning rate slightly to settle
            lr_curr *= 0.99
            
        # Check if valid
        current_r = calc_radius(centers)
        
        if current_r >= r - 1e-5:
            # We successfully maintained radius r, try to increase it
            r += dr
            step += max_iter_per_r
        else:
            # Failed to maintain radius, break
            break
            
    # Final Refinement: Local Search focusing on closest pairs
    # The relaxation might leave the system in a state where it's "stable" but not optimal.
    # We can try to optimize by moving spheres to increase the minimum distance.
    
    r_best = calc_radius(centers)
    
    # Simple Hill Climbing with more aggressive moves and multiple attempts
    # Since N=100, O(N^2) distance calc is fast.
    
    for _ in range(10000):
        i = np.random.randint(N)
        # Try a random move
        delta = np.random.normal(0, 0.005, 3)
        new_c = centers.copy()
        new_c[i] = np.clip(centers[i] + delta, -0.5, 0.5)
        new_r = calc_radius(new_c)
        if new_r > r_best + 1e-8:
            centers = new_c
            r_best = new_r
        else:
            # Try a smaller move if it failed? Or just continue.
            pass
            
    return centers
