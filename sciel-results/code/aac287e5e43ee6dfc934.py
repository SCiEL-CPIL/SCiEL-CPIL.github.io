import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    def score(centers):
        # Calculate min distance between centers
        # and min distance to walls
        # Return the minimum of these distances / 2 (radius)
        # Actually, the score is r.
        # r = min( min_{i<j} ||c_i - c_j|| / 2, min_i min_k (0.5 - |c_ik|) )
        
        # Efficient calculation of min pairwise distance
        # For N=50, O(N^2) is 2500 ops, very fast.
        min_dist = 1.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < min_dist:
                    min_dist = d
        
        # Wall distance
        # centers are in [-0.5, 0.5]
        # dist to wall = 0.5 - max(abs(x), abs(y)) ? No, min distance to any wall.
        # min(0.5 - x, 0.5 + x, 0.5 - y, 0.5 + y)
        # which is 0.5 - max(abs(x), abs(y))
        wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
        min_wall_dist = np.min(wall_dists)
        
        r = min(min_dist / 2.0, min_wall_dist)
        return r

    def ls_algorithm(centers, r_start=0.05, max_time=100):
        """
        Lubachevsky-Stillinger style relaxation
        """
        centers = centers.copy()
        r = r_start
        dt = 0.01
        damping = 0.9
        velocities = np.zeros_like(centers)
        
        # Pre-calculate grid spacing for initial r estimate if needed
        # But we pass r_start
        
        while time.time() < max_time:
            # Calculate forces
            forces = np.zeros_like(centers)
            
            # Pairwise repulsion
            for i in range(N):
                for j in range(i+1, N):
                    d_vec = centers[i] - centers[j]
                    d = np.linalg.norm(d_vec)
                    if d < 2.0 * r:
                        # Overlap
                        if d > 1e-6:
                            # Force proportional to overlap, pushing apart
                            # F = k * (2r - d) * d_vec / d
                            # Let's use a strong spring constant
                            k = 1.0 
                            f_mag = k * (2.0 * r - d)
                            f_vec = f_mag * d_vec / d
                            forces[i] += f_vec
                            forces[j] -= f_vec
                        else:
                            # Random kick
                            forces[i] += np.array([np.random.randn(), np.random.randn()])
                            forces[j] -= np.array([np.random.randn(), np.random.randn()])
                            
            # Wall repulsion
            for i in range(N):
                x, y = centers[i]
                # Left wall
                if x < -0.5 + r:
                    forces[i][0] += 1.0 * (-0.5 + r - x)
                # Right wall
                if x > 0.5 - r:
                    forces[i][0] -= 1.0 * (x - (0.5 - r))
                # Bottom wall
                if y < -0.5 + r:
                    forces[i][1] += 1.0 * (-0.5 + r - y)
                # Top wall
                if y > 0.5 - r:
                    forces[i][1] -= 1.0 * (y - (0.5 - r))
            
            # Update velocities and positions
            velocities = damping * velocities + forces * dt
            centers += velocities * dt
            
            # Ensure centers stay within bounds (hard constraint)
            # This helps stability
            centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
            centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
            
            # Check if we can increase r
            # If no overlaps, increase r
            # Simple check: min distance > 2r + epsilon
            min_d = 1.0
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(centers[i] - centers[j])
                    if d < min_d:
                        min_d = d
            
            # Also check walls
            min_wall = np.min(0.5 - np.max(np.abs(centers), axis=1))
            
            current_r_max = min(min_d / 2.0, min_wall)
            
            if current_r_max > r * (1.0 + 1e-4):
                # Increase r towards the max possible, but slowly
                # To avoid instability, increase by a small factor
                r = r * 1.001 + 1e-5
                if r > current_r_max:
                    r = current_r_max
            else:
                # Relax
                pass

        return centers, r

    # Initial Guess: Grid
    m = int(np.ceil(np.sqrt(N)))
    # Use a grid that fits in the square
    # Side length 1. m circles per side.
    # Spacing 1/m.
    # Radius approx 1/(2m)
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # Calculate initial radius for grid
    # For a grid, min dist is 1/m. r = 1/(2m).
    # But corners are closer to walls? No, grid is centered.
    # r_grid = 1/(2*m)
    r_init = 1.0 / (2.0 * m) * 0.9 # Start slightly smaller to allow relaxation
    
    best_centers = centers
    best_r = score(centers)
    
    # Run LS algorithm with a time limit
    # We can try multiple restarts if time permits
    # But LS is usually good from a grid start.
    
    # Let's run LS
    centers_ls, r_ls = ls_algorithm(centers, r_start=r_init, max_time=time.time() + 90)
    
    r_final = score(centers_ls)
    if r_final > best_r:
        best_centers = centers_ls
        best_r = r_final
        
    # If time remains, try a local optimization polish
    # Using scipy.optimize.minimize with Nelder-Mead on the negative score
    if time.time() < deadline - 10:
        try:
            from scipy.optimize import minimize
            
            def neg_score(x):
                c = x.reshape(N, 2)
                return -score(c)
            
            # Use Nelder-Mead
            res = minimize(neg_score, best_centers.flatten(), method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if res.success or score(res.x.reshape(N,2)) > best_r:
                c_opt = res.x.reshape(N, 2)
                r_opt = score(c_opt)
                if r_opt > best_r:
                    best_centers = c_opt
                    best_r = r_opt
        except Exception as e:
            pass
            
    return best_centers

# EVOLVE-BLOCK-END
