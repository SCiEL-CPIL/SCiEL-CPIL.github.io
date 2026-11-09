import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    def exact_radius(C):
        # Compute min pairwise distance / 2
        # and min distance to boundary
        diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pairwise = np.min(dists) / 2.0
        
        # Boundary
        d_b = 0.5 - np.abs(C)
        min_boundary = np.min(d_b)
        
        return min(min_pairwise, min_boundary)

    def soft_min_obj_and_grad(C_flat, beta):
        C = C_flat.reshape(N, 3)
        
        # Pairwise
        diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        i_idx, j_idx = np.triu_indices(N, k=1)
        pair_dists = dists[i_idx, j_idx] / 2.0
        
        # Boundary
        d_b = 0.5 - np.abs(C)
        boundary_dists = d_b.flatten()
        
        all_dists = np.concatenate([pair_dists, boundary_dists])
        
        # Soft min: -1/beta * log(sum(exp(-beta * x)))
        y = -beta * all_dists
        max_y = np.max(y)
        if not np.isfinite(max_y):
            return 0.0, np.zeros_like(C_flat)
        
        exp_y = np.exp(y - max_y)
        sum_exp = np.sum(exp_y)
        log_sum = max_y + np.log(sum_exp)
        
        obj = -1.0 / beta * log_sum
        
        # Gradient
        w = exp_y / sum_exp
        dw_dx = w / beta
        
        grad_C = np.zeros_like(C)
        
        # Pairwise
        w_pair = dw_dx[:len(pair_dists)]
        pair_full_dists = pair_dists * 2.0
        pair_full_dists[pair_full_dists < 1e-10] = 1e-10
        
        for k in range(len(pair_dists)):
            i = i_idx[k]
            j = j_idx[k]
            d = pair_full_dists[k]
            vec = (C[i] - C[j]) / d
            grad_C[i] += w_pair[k] * vec
            grad_C[j] -= w_pair[k] * vec
            
        # Boundary
        w_bound = dw_dx[len(pair_dists):].reshape(N, 3)
        signs = np.sign(C)
        signs[signs == 0] = 1.0
        grad_C -= w_bound * signs
        
        return obj, grad_C.flatten()

    best_C = None
    best_radius = -1.0

    # Multi-start
    num_restarts = 10
    np.random.seed(12345)
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10.0: # Leave time for final checks
            break
            
        # Initialization: Sample 50 points from a 5x5x5 grid
        # Grid points in [-0.4, 0.4] to stay safely inside initially?
        # No, let's use [-0.5, 0.5] but jitter will move them.
        # 5x5x5 = 125 points.
        grid_coords = np.linspace(-0.45, 0.45, 5)
        grid_points = np.array(np.meshgrid(grid_coords, grid_coords, grid_coords)).T.reshape(-1, 3)
        
        # Shuffle and take 50
        np.random.shuffle(grid_points)
        C = grid_points[:N].copy()
        
        # Add jitter
        C += np.random.uniform(-0.02, 0.02, C.shape)
        
        # Clamp to ensure they are inside the cube for the first step
        # The optimizer will handle the rest, but starting inside is good.
        C = np.clip(C, -0.49, 0.49)
        
        current_best_C = C.copy()
        current_best_r = exact_radius(current_best_C)
        
        # Anneal beta
        betas = [10, 50, 200, 1000]
        
        for beta in betas:
            if time.time() > deadline - 5.0:
                break
                
            try:
                res = minimize(
                    soft_min_obj_and_grad, 
                    current_best_C.flatten(), 
                    args=(beta,), 
                    method='L-BFGS-B', 
                    jac=True,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
                )
                C_new = res.x.reshape(N, 3)
                
                if not np.all(np.isfinite(C_new)):
                    continue
                    
                r_new = exact_radius(C_new)
                
                if r_new > current_best_r:
                    current_best_r = r_new
                    current_best_C = C_new
                    
            except Exception:
                pass
        
        if current_best_r > best_radius:
            best_radius = current_best_r
            best_C = current_best_C

    # Final Polish: Coordinate Descent on exact radius
    # This helps if the soft-min left small errors
    if best_C is not None and time.time() < deadline - 5.0:
        C_polish = best_C.copy()
        improved = True
        step = 0.01
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in range(N):
                for axis in range(3):
                    for direction in [-1, 1]:
                        trial = C_polish.copy()
                        trial[i, axis] += direction * step
                        # Check bounds
                        if np.any(trial < -0.5) or np.any(trial > 0.5):
                            continue
                        r_trial = exact_radius(trial)
                        if r_trial > exact_radius(C_polish):
                            C_polish = trial
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                step *= 0.5
                if step < 1e-5:
                    break
        
        r_polish = exact_radius(C_polish)
        if r_polish > best_radius:
            best_radius = r_polish
            best_C = C_polish

    return best_C
