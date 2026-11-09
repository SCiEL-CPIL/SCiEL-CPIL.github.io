"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Greedy Initialization
    # Place spheres one by one to maximize the minimum distance to existing spheres and walls.
    # This is O(N^3) but N=50 is small.
    def greedy_init():
        centers = np.zeros((N, 3))
        # Start with center
        centers[0] = [0.0, 0.0, 0.0]
        
        for i in range(1, N):
            best_pos = None
            best_min_dist = -1.0
            # Sample candidate positions
            # Use a grid of candidates to speed up, or random samples
            # A grid of 10x10x10 = 1000 points is fast enough.
            grid_size = 12
            ticks = np.linspace(-0.45, 0.45, grid_size)
            candidates = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
            
            for c in candidates:
                # Check if inside cube (with margin for radius, but radius is unknown yet, 
                # so we just ensure it's well inside, e.g. 0.1 from walls)
                # Actually, we just want to maximize min dist to existing and walls.
                min_dist_wall = 0.5 - np.max(np.abs(c))
                min_dist_spheres = np.min(np.linalg.norm(centers[:i] - c, axis=1)) if i > 0 else 1.0
                current_min = min(min_dist_wall, min_dist_spheres)
                
                if current_min > best_min_dist:
                    best_min_dist = current_min
                    best_pos = c
            
            if best_pos is not None:
                centers[i] = best_pos
            else:
                # Fallback to random if something went wrong
                centers[i] = np.random.uniform(-0.4, 0.4, 3)
                
        return centers

    def objective(params, beta):
        # params: flattened array of shape (N*3,)
        c = params.reshape(N, 3)
        
        # Calculate pairwise distances
        # diff: (N, N, 3)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Zero out diagonal
        np.fill_diagonal(dists, 1e10)
        
        # Pair terms: exp(-beta * dist)
        # We only need upper triangle to avoid double counting, but for log-sum-exp 
        # symmetry doesn't matter much for the gradient, just the value. 
        # However, double counting weights the pair distance twice. 
        # It's better to use unique pairs.
        iu = np.triu_indices(N, k=1)
        pair_dists = dists[iu]
        
        # Wall distances
        # c: (N, 3)
        # dist to x-walls: 0.5 - abs(c[:, 0])
        # dist to y-walls: 0.5 - abs(c[:, 1])
        # dist to z-walls: 0.5 - abs(c[:, 2])
        wall_dists = np.concatenate([
            0.5 - np.abs(c[:, 0]),
            0.5 - np.abs(c[:, 1]),
            0.5 - np.abs(c[:, 2])
        ])
        
        # Combine
        # We want to minimize the log-sum-exp of exp(-beta * dist)
        # To avoid overflow/underflow issues with large beta:
        # log(sum(exp(-beta*d))) = -beta*max(d) + log(sum(exp(-beta*(d-max(d)))))
        # Wait, we are minimizing. 
        # If d is small, exp(-beta*d) is large.
        # Let's just compute directly if beta is not too large.
        # If beta is large, use the shift trick.
        
        all_dists = np.concatenate([pair_dists, wall_dists])
        
        # Shift for numerical stability
        d_max = np.max(all_dists)
        shifted = -beta * (all_dists - d_max)
        
        # Clip shifted values to prevent exp overflow (though shifted <= 0, so exp <= 1)
        # Actually, if all_dists are positive, -beta*(d-dmax) is <= 0.
        # So exp is in (0, 1]. Sum is <= count. Log is safe.
        
        sum_exp = np.sum(np.exp(shifted))
        
        # The objective value is (1/beta) * log(sum_exp) + d_max
        # Wait, log(sum(exp(-beta*d))) = -beta*d_max + log(sum(exp(-beta*(d-d_max))))
        # We want to minimize this quantity.
        # As beta -> inf, this approaches min(d).
        # So we minimize this value.
        
        val = d_max + (1.0 / beta) * np.log(sum_exp)
        
        return val

    def objective_wrapper(params, beta):
        return objective(params, beta)

    best_centers = greedy_init()
    best_score = -1.0 # We are minimizing the objective, but we want to track the actual radius
    
    def calc_radius(centers):
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 1e10)
        min_pair = np.min(dists)
        
        wall_dists = np.concatenate([
            0.5 - np.abs(centers[:, 0]),
            0.5 - np.abs(centers[:, 1]),
            0.5 - np.abs(centers[:, 2])
        ])
        min_wall = np.min(wall_dists)
        
        return min(min_pair / 2.0, min_wall)

    # Optimization Loop with Annealing
    # Try a few different initializations if time permits
    num_restarts = 5
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        if r == 0:
            init_centers = best_centers
        else:
            # Jitter the greedy init
            init_centers = best_centers + np.random.normal(0, 0.05, (N, 3))
            # Clip to valid range
            init_centers = np.clip(init_centers, -0.49, 0.49)
            
        current_centers = init_centers.copy()
        
        # Annealing schedule
        betas = [10, 50, 200, 1000, 5000]
        
        for beta in betas:
            if time.time() > deadline:
                break
                
            params = current_centers.flatten()
            
            # Bounds
            bounds = [(-0.5, 0.5)] * (N * 3)
            
            try:
                res = minimize(
                    objective_wrapper,
                    params,
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if res.success:
                    new_centers = res.x.reshape(N, 3)
                    # Check if it's valid (inside cube)
                    if np.all(new_centers >= -0.5) and np.all(new_centers <= 0.5):
                        current_centers = new_centers
            except Exception:
                pass
        
        # Evaluate current best
        radius = calc_radius(current_centers)
        if radius > best_score:
            best_score = radius
            best_centers = current_centers.copy()
            
    return best_centers
# EVOLVE-BLOCK-END
