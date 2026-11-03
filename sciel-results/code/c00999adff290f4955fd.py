import numpy as np
import time
from scipy.optimize import minimize

N = 100
CUBE_HALF = 0.5

def _soft_min_distance(centers, T):
    """
    Compute a differentiable approximation of the minimum distance.
    centers: (N, 3) array
    T: temperature parameter for soft-min
    Returns: -soft_min_dist (to be minimized)
    """
    # Compute pairwise distances
    # diff[i, j] = centers[i] - centers[j]
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    # Exclude self-distances (diagonal)
    np.fill_diagonal(dist_sq, np.inf)
    
    # Compute distances
    dist = np.sqrt(dist_sq)
    
    # Soft min: -T * log(sum(exp(-dist/T)))
    # To avoid underflow/overflow, we use log-sum-exp trick
    # min_val = -T * (log(sum(exp(-dist/T))) - min(-dist/T))
    # Actually, simpler:
    # f = -T * logsumexp(-dist/T)
    
    # We want to maximize min(dist), so we minimize -min(dist).
    # Let g = -T * log(sum(exp(-dist/T)))
    # As T -> 0, g -> -min(dist).
    
    # For numerical stability:
    # logsumexp(x) = max(x) + log(sum(exp(x - max(x))))
    # Here x = -dist/T. max(x) = -min(dist)/T.
    # This is getting complicated. Let's just use the standard log-sum-exp.
    
    x = -dist / T
    # Mask invalid (inf) values
    x = x[np.isfinite(x)]
    
    max_x = np.max(x)
    if not np.isfinite(max_x):
        return 0.0
        
    sum_exp = np.sum(np.exp(x - max_x))
    log_sum_exp = max_x + np.log(sum_exp)
    
    # The value we want to minimize is - (approx min dist)
    # g = -T * log_sum_exp
    # Note: g is negative. Minimizing g (making it more negative) increases the min dist.
    # Wait. 
    # min(dist) = -T * log(sum(exp(-dist/T)))
    # So we want to MAXIMIZE min(dist).
    # So we MINIMIZE -min(dist) = T * log(sum(exp(-dist/T)))
    
    return T * log_sum_exp

def _compute_score(centers):
    """
    Compute the actual score: largest common radius.
    """
    if centers.shape != (N, 3):
        return 0.0
    
    # Check boundaries
    if np.any(centers < -CUBE_HALF) or np.any(centers > CUBE_HALF):
        return 0.0
        
    # Check finite
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_sq = np.min(dist_sq)
    min_dist = np.sqrt(min_dist_sq)
    
    # Distance to walls
    # min_i min_k (1/2 - |c_ik|)
    # Since centers are in [-0.5, 0.5], distance to wall is min(0.5 - x, x + 0.5) = 0.5 - |x|
    dist_to_wall = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall_dist = np.min(dist_to_wall)
    
    r = min(min_dist / 2.0, min_wall_dist)
    
    if r <= 0 or not np.isfinite(r):
        return 0.0
        
    return r

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin for 150s limit
    
    best_centers = None
    best_score = 0.0
    
    # Initial guess: Random points
    # We will try a few restarts with different initializations and annealing schedules
    
    # Grid-based initialization is often a good starting point
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    init_centers = grid[:N].copy()
    
    # Add small random jitter to break symmetry
    init_centers += np.random.normal(0, 0.01, size=init_centers.shape)
    # Clip to bounds
    init_centers = np.clip(init_centers, -CUBE_HALF + 1e-6, CUBE_HALF - 1e-6)
    
    # Annealing schedule: Start with larger T (smoother), end with smaller T (sharper)
    # T controls the "softness" of the minimum.
    # Large T: objective is dominated by many pairs, easier to optimize globally.
    # Small T: objective is dominated by the closest pair, harder but more precise.
    
    temperatures = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
    
    current_centers = init_centers.copy()
    
    for T in temperatures:
        if time.time() - start_time > timeout:
            break
            
        # Define objective and gradient for L-BFGS-B
        def objective(x_flat):
            centers = x_flat.reshape(N, 3)
            return _soft_min_distance(centers, T)
            
        # Numerical gradient is safer if analytical is complex, but L-BFGS-B can do finite diff.
        # For 300 variables, finite diff is okay if not too many iterations.
        # However, we can provide a numerical gradient hint or just let scipy do it.
        # Let's let scipy handle the gradient with finite differences.
        
        bounds = [(-CUBE_HALF + 1e-6, CUBE_HALF - 1e-6)] * (N * 3)
        
        try:
            res = minimize(
                objective,
                current_centers.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
            )
            
            if res.success:
                current_centers = res.x.reshape(N, 3)
                # Clip just in case
                current_centers = np.clip(current_centers, -CUBE_HALF, CUBE_HALF)
                
                score = _compute_score(current_centers)
                if score > best_score:
                    best_score = score
                    best_centers = current_centers.copy()
                    
        except Exception as e:
            pass
            
    # Final refinement with very small T if time permits
    if time.time() - start_time < timeout - 5:
        T_final = 0.001
        def objective_final(x_flat):
            centers = x_flat.reshape(N, 3)
            return _soft_min_distance(centers, T_final)
            
        bounds = [(-CUBE_HALF + 1e-6, CUBE_HALF - 1e-6)] * (N * 3)
        
        try:
            res = minimize(
                objective_final,
                current_centers.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False}
            )
            
            if res.success:
                final_centers = res.x.reshape(N, 3)
                final_centers = np.clip(final_centers, -CUBE_HALF, CUBE_HALF)
                
                score = _compute_score(final_centers)
                if score > best_score:
                    best_score = score
                    best_centers = final_centers.copy()
                    
        except Exception as e:
            pass
            
    # If no valid best_centers found (should not happen), return grid
    if best_centers is None:
        return init_centers
        
    return best_centers

# EVOLVE-BLOCK-END
