import numpy as np
import time
from scipy.optimize import minimize

N = 200

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit

    def compute_radius(pos):
        """Compute the maximum radius for a given configuration."""
        pos = np.asarray(pos)
        # Wall distances
        wall_dists = 0.5 - np.max(np.abs(pos), axis=1)
        min_wall = np.min(wall_dists)
        
        # Pairwise distances
        # Use a trick to avoid computing full NxN if possible, but N=200 is small
        D = np.linalg.norm(pos[:, np.newaxis, :] - pos[np.newaxis, :, :], axis=2)
        np.fill_diagonal(D, np.inf)
        min_pair = np.min(D)
        
        return min(min_wall, min_pair / 2.0)

    def initial_grid():
        m = int(np.ceil(np.sqrt(N)))
        # Try to fit into square grid
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        pos = grid[:N].copy()
        # Add small random noise
        pos += np.random.normal(0, 0.001, pos.shape)
        # Clamp to valid region
        pos = np.clip(pos, -0.45, 0.45)
        return pos

    def random_start():
        # Random points in [-0.4, 0.4]
        pos = np.random.uniform(-0.4, 0.4, (N, 2))
        return pos

    best_pos = None
    best_r = 0.0

    # Strategy 1: Start from Grid and run SA
    # Strategy 2: Run SLSQP on the best result from SA
    
    # Phase 1: Simulated Annealing (fast, robust)
    pos = initial_grid()
    D = np.linalg.norm(pos[:, np.newaxis, :] - pos[np.newaxis, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    wall_dists = 0.5 - np.max(np.abs(pos), axis=1)
    
    current_r = min(np.min(D) / 2.0, np.min(wall_dists))
    best_pos = pos.copy()
    best_r = current_r

    T = 0.005
    T_min = 1e-5
    alpha = 0.999
    base_step = 0.01

    steps = 0
    while time.time() - start_time < time_limit - 20.0:  # Leave 20s for SLSQP
        i = np.random.randint(N)
        step_size = base_step * (1.0 + T / 0.005)
        delta = np.random.normal(0, step_size, 2)
        
        new_pos = pos.copy()
        new_pos[i] += delta
        
        # Clamp to square
        new_pos[i] = np.clip(new_pos[i], -0.5, 0.5)
        
        # Update D for row/col i
        new_dists_i = np.linalg.norm(new_pos[i] - new_pos, axis=1)
        D[i, :] = new_dists_i
        D[:, i] = new_dists_i
        D[i, i] = np.inf
        
        # Update wall dists for i
        new_wall_dist_i = 0.5 - np.max(np.abs(new_pos[i]))
        wall_dists[i] = new_wall_dist_i
        
        new_min_pair = np.min(D)
        new_min_wall = np.min(wall_dists)
        new_r = min(new_min_pair / 2.0, new_min_wall)
        
        if new_r > current_r:
            pos = new_pos
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_pos = pos.copy()
        else:
            delta_r = current_r - new_r
            if delta_r > 0:
                prob = np.exp(-delta_r / T)
                if np.random.random() < prob:
                    pos = new_pos
                    current_r = new_r
        
        T *= alpha
        if T < T_min:
            T = T_min
            # Restart if stuck
            if best_r < 0.0366: # Far from known best
                T = 0.005
                # Perturb current best
                pos = best_pos + np.random.normal(0, 0.005, best_pos.shape)
                pos = np.clip(pos, -0.5, 0.5)
                D = np.linalg.norm(pos[:, np.newaxis, :] - pos[np.newaxis, :, :], axis=2)
                np.fill_diagonal(D, np.inf)
                wall_dists = 0.5 - np.max(np.abs(pos), axis=1)
                current_r = min(np.min(D) / 2.0, np.min(wall_dists))
        
        steps += 1

    # Phase 2: SLSQP Optimization
    # We try to maximize the minimum distance.
    # SLSQP minimizes, so we minimize -r.
    # Constraints:
    #   g1_i: d_ij - 2r >= 0 for all i<j  -> d_ij - 2r
    #   g2_i: 0.5 - |x_i| - r >= 0
    #   g3_i: 0.5 - |y_i| - r >= 0
    
    # This is complex because r is a variable.
    # Alternative: Fix r and check feasibility? No, we want to find max r.
    # Let's optimize x directly with a penalty or constraints that don't depend on r?
    # No, the objective is to maximize r.
    
    # Let's use a different formulation:
    # Minimize -min(d_ij/2, wall_i)
    # This is non-smooth.
    
    # Better: Use SLSQP to maximize the minimum pairwise distance, and separately ensure wall constraints.
    # Or, include r as a variable.
    
    # Variables: x_flat (400 values)
    # Objective: minimize -r
    # Constraints:
    #   For all pairs i<j: ||x_i - x_j|| - 2r >= 0
    #   For all i: 0.5 - |x_ik| - r >= 0
    
    # This has ~ 400 + 2*200*199/2 + 4*200 constraints. That's a lot for SLSQP.
    # N=200, pairs ~ 20000.
    # SLSQP might be slow.
    
    # Let's try a local optimization using gradient descent on the smooth approximation or just a simpler gradient ascent on the radius.
    
    # Alternative Phase 2: Gradient Ascent on the objective f(x) = min( min_pair/2, min_wall )
    # This is non-differentiable, but we can use subgradients or just coordinate descent.
    
    # Let's stick to a robust "Push Apart" local optimization which is fast.
    
    final_pos = best_pos.copy()
    D_final = np.linalg.norm(final_pos[:, np.newaxis, :] - final_pos[np.newaxis, :, :], axis=2)
    np.fill_diagonal(D_final, np.inf)
    
    for _ in range(2000):
        if time.time() - start_time > time_limit - 5.0:
            break
        
        # Find closest pair
        min_val = np.min(D_final)
        min_idx = np.unravel_index(np.argmin(D_final), D_final.shape)
        i, j = min_idx
        
        # Push apart
        vec = final_pos[i] - final_pos[j]
        norm_vec = np.linalg.norm(vec)
        if norm_vec < 1e-9:
            vec = np.array([1.0, 0.0])
            norm_vec = 1.0
        else:
            vec /= norm_vec
            
        # Move them apart
        push = 0.0005
        final_pos[i] += vec * push
        final_pos[j] -= vec * push
        
        # Clamp to square
        final_pos = np.clip(final_pos, -0.5, 0.5)
        
        # Update D
        D_final[i, :] = np.linalg.norm(final_pos[i] - final_pos, axis=1)
        D_final[:, i] = D_final[i, :]
        D_final[j, :] = np.linalg.norm(final_pos[j] - final_pos, axis=1)
        D_final[:, j] = D_final[j, :]
        D_final[i, i] = np.inf
        D_final[j, j] = np.inf
        
        # Also check walls: if a point is close to wall, push it in
        # Find point with min wall distance
        wall_dists_final = 0.5 - np.max(np.abs(final_pos), axis=1)
        min_wall_idx = np.argmin(wall_dists_final)
        p = final_pos[min_wall_idx]
        
        # Direction to center
        # Gradient of distance to nearest wall
        # If x > 0, wall is at 0.5, dist is 0.5-x. Gradient wrt x is -1. To increase dist, decrease x.
        # So move towards 0.
        
        # Check if wall constraint is tighter than pair constraint
        # r_pair = D_final[i,j]/2
        # r_wall = 0.5 - max(abs(p))
        
        # If we just push the closest pair, we might violate walls.
        # The clamp handles invalid states, but it might reduce the radius.
        # Let's calculate the new radius.
        
    # Calculate final scores
    r_best = best_r
    r_final = compute_radius(final_pos)
    
    if r_final > r_best:
        return final_pos
    else:
        return best_pos

