"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    # Initial guess: Perturbed Grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    rng = np.random.default_rng(42)
    
    alpha = 50.0
    best_result = None
    best_obj = np.inf
    
    # Run multiple optimizations with different perturbations
    for i in range(10):
        centers_init = grid[:N].copy()
        # Increase perturbation magnitude slightly for later runs
        scale = 0.01 * (1 + i * 0.5)
        centers_init += rng.normal(0, scale, size=centers_init.shape)
        centers_init = np.clip(centers_init, -0.49, 0.49)
        
        def objective_and_grad(x):
            c = x.reshape(N, 3)
            
            # Pairwise distances
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            
            iu = np.triu_indices(N, 1)
            pair_dists = dists[iu]
            
            # Wall distances
            wall_dists = np.concatenate([
                0.5 - c[:, 0],
                0.5 + c[:, 0],
                0.5 - c[:, 1],
                0.5 + c[:, 1],
                0.5 - c[:, 2],
                0.5 + c[:, 2]
            ])
            
            all_dists = np.concatenate([pair_dists, wall_dists])
            
            max_d = np.max(all_dists)
            shifted = -alpha * (all_dists - max_d)
            shifted = np.clip(shifted, -100, 100)
            
            exp_shifted = np.exp(shifted)
            sum_exp = np.sum(exp_shifted)
            
            if sum_exp == 0:
                return 0, np.zeros_like(x)
                
            weights = exp_shifted / sum_exp
            
            obj = max_d + np.log(sum_exp)
            
            # Gradient calculation
            pair_weights = weights[:len(pair_dists)]
            
            w_mat = np.zeros((N, N))
            w_mat[iu] = pair_weights
            w_mat.T[iu] = pair_weights
            
            safe_dists = np.where(dists > 1e-8, dists, 1.0)
            
            F = w_mat[:, :, np.newaxis] * diff / safe_dists[:, :, np.newaxis]
            grad_pairs = np.sum(F, axis=1)
            
            wall_weights = weights[len(pair_dists):]
            grad_walls = np.zeros((N, 3))
            
            grad_walls[:, 0] += (-wall_weights[0::6] + wall_weights[1::6])
            grad_walls[:, 1] += (-wall_weights[2::6] + wall_weights[3::6])
            grad_walls[:, 2] += (-wall_weights[4::6] + wall_weights[5::6])
            
            grad = grad_pairs + grad_walls
            
            # We are minimizing obj.
            # d(obj)/dx = -alpha * grad
            return obj, -alpha * grad

        res = minimize(objective_and_grad, centers_init.flatten(), method='L-BFGS-B', 
                       jac=True, options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res.fun < best_obj:
            best_obj = res.fun
            best_result = res.x
    
    best_centers = best_result.reshape(N, 3)
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
# EVOLVE-BLOCK-END
