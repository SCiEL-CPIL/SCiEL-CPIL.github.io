import numpy as np
import time
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Initial Grid Construction
    # 5x5x4 grid = 100 points
    # Cube is [-0.5, 0.5]
    # Spacing in x, y is 1/5 = 0.2. In z is 1/4 = 0.25.
    # Points are centered.
    # x, y: -0.4, -0.2, 0, 0.2, 0.4
    # z: -0.375, -0.125, 0.125, 0.375
    
    xs = np.linspace(-0.4, 0.4, 5)
    ys = np.linspace(-0.4, 0.4, 5)
    zs = np.linspace(-0.375, 0.375, 4)
    
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    
    def calc_radius(centers):
        # Min distance between centers / 2
        # Min distance to boundary
        if len(centers) < 2:
            return 0.5
        # Pairwise distances
        # Using broadcasting for vectorization
        # diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        # dists = np.sqrt(np.sum(diff**2, axis=2))
        # np.fill_diagonal(dists, np.inf)
        # min_dist = np.min(dists)
        
        # More efficient:
        # We only need min.
        # But for 100 points, 100x100 matrix is fine.
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        # Boundary distance
        # min_k (0.5 - |c_ik|)
        abs_c = np.abs(centers)
        min_bound = np.min(0.5 - abs_c)
        
        return min(min_dist / 2.0, min_bound)

    best_centers = grid
    best_r = calc_radius(grid)
    
    # Theoretical upper bound check
    # Volume argument: 100 * (4/3) pi r^3 <= 1  => r <= (3 / (400 pi))^(1/3) ~ 0.133
    # Best known is 0.11138.
    # Let's start bisection between best_r and 0.12
    
    r_low = best_r
    r_high = 0.12
    
    # Pre-compute indices for penalty calculation to avoid overhead
    # i < j
    idx_i, idx_j = np.triu_indices(N, k=1)
    # Number of pairs is ~5000
    
    def penalty_func(x_flat, r_target):
        # x_flat is shape (300,)
        centers = x_flat.reshape(N, 3)
        
        # 1. Overlap Penalty
        # diff = centers[idx_i] - centers[idx_j]
        # dists = np.sqrt(np.sum(diff**2, axis=1))
        # overlaps = np.maximum(0, 2*r_target - dists)
        # pen_overlap = np.sum(overlaps**2)
        
        diff = centers[idx_i] - centers[idx_j]
        dists_sq = np.sum(diff**2, axis=1)
        # Avoid sqrt if possible? No, we need dists for the max.
        # But max(0, 2r - d)^2 is smooth enough.
        dists = np.sqrt(dists_sq)
        overlaps = np.maximum(0, 2*r_target - dists)
        pen_overlap = np.sum(overlaps**2)
        
        # 2. Boundary Penalty
        # centers shape (N, 3)
        # limits = 0.5 - r_target
        limit = 0.5 - r_target
        # viol_pos = np.maximum(0, centers - limit)
        # viol_neg = np.maximum(0, -limit - centers)
        # pen_bound = np.sum(viol_pos**2) + np.sum(viol_neg**2)
        
        viol_pos = np.maximum(0, centers - limit)
        viol_neg = np.maximum(0, -limit - centers)
        pen_bound = np.sum(viol_pos**2) + np.sum(viol_neg**2)
        
        return pen_overlap + pen_bound

    def try_optimize(r_target, initial_guess, max_iter=500):
        # Use L-BFGS-B
        # Bounds: each coordinate in [-0.5, 0.5]
        # Actually, we can tighten bounds to [-0.5+r, 0.5-r] to help?
        # If we use bounds, the penalty for boundary becomes zero if bounds are respected.
        # But if we set bounds based on r_target, and the optimizer hits the bound,
        # it means the sphere is touching the wall, which is fine.
        # However, if the required radius is larger than the space allows for that config,
        # it will fail.
        
        bounds = [(-0.5, 0.5) for _ in range(3*N)]
        
        # Initial guess flattened
        x0 = initial_guess.flatten()
        
        res = minimize(
            penalty_func, 
            x0, 
            args=(r_target,), 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        return res

    # Bisection Loop
    # We will try a few random restarts for each r_target to escape local minima
    # Time budget per r_target: ~5 seconds?
    # 110s / 5s = 22 iterations.
    # Each iteration: 5 restarts * 500 iters? Might be too slow.
    # Let's reduce max_iter to 200 and restarts to 3.
    
    # To speed up, we can use the result of the previous successful r as the initial guess for the next.
    current_guess = grid
    
    while time.time() < deadline:
        r_mid = (r_low + r_high) / 2.0
        
        # Try to find a valid config for r_mid
        found_valid = False
        best_res_for_r = None
        best_penalty_for_r = np.inf
        
        # Try 3 restarts: 1 from current best, 2 random perturbations
        guesses = [current_guess]
        # Add small noise to current guess
        noise = np.random.normal(0, 0.01, (N, 3))
        guesses.append(np.clip(current_guess + noise, -0.5, 0.5))
        # Random grid-like start
        guesses.append(np.random.uniform(-0.45, 0.45, (N, 3)))
        
        for g in guesses:
            if time.time() > deadline - 5: break # Safety
            
            res = try_optimize(r_mid, g, max_iter=300)
            if res.fun < best_penalty_for_r:
                best_penalty_for_r = res.fun
                best_res_for_r = res
                
            if best_penalty_for_r < 1e-8:
                found_valid = True
                break
        
        if found_valid:
            r_low = r_mid
            # Update current guess to the found solution
            if best_res_for_r is not None:
                current_guess = best_res_for_r.x.reshape(N, 3)
                # Verify radius
                r_check = calc_radius(current_guess)
                if r_check >= r_mid - 1e-5:
                    best_centers = current_guess
                    best_r = r_check
        else:
            r_high = r_mid
            
        # If the gap is small, break
        if r_high - r_low < 1e-6:
            break

    return best_centers
