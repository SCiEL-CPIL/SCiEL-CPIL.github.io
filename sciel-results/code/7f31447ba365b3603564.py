import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0  # Leave 10s margin

    def compute_soft_min(points, beta=10.0):
        # points shape (N, 3)
        # Pairwise
        dists = np.linalg.norm(points[:, np.newaxis] - points[np.newaxis, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        pair_dists = dists / 2.0
        
        # Wall
        wall_dists = 0.5 - np.abs(points)
        
        all_dists = np.concatenate([pair_dists.flatten(), wall_dists.flatten()])
        
        m = np.min(all_dists)
        if not np.isfinite(m):
            return -1e6, 0.0 # Return very negative soft-min (bad) and 0 gradient? 
                             # Actually, if m is -inf, something is wrong.
        
        exp_terms = np.exp(-beta * (all_dists - m))
        sum_exp = np.sum(exp_terms)
        if sum_exp == 0:
            return -1e6, 0.0
            
        soft_min_val = m - (1.0 / beta) * np.log(sum_exp)
        return soft_min_val, sum_exp

    def objective_and_grad(points_flat):
        pts = points_flat.reshape(N, 3)
        
        # Compute distances
        dists = np.linalg.norm(pts[:, np.newaxis] - pts[np.newaxis, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        pair_dists = dists / 2.0
        
        wall_dists = 0.5 - np.abs(pts)
        
        # Weights for soft-min
        beta = 10.0
        
        # Combine for calculation
        # To get gradient, we need weights w_k = exp(-beta * d_k) / sum(exp(-beta * d_j))
        
        # Shift by min for stability
        m = np.min(np.concatenate([pair_dists.flatten(), wall_dists.flatten()]))
        
        # Calculate exps
        exp_pairs = np.exp(-beta * (pair_dists - m))
        exp_walls = np.exp(-beta * (wall_dists - m))
        
        sum_exp = np.sum(exp_pairs) + np.sum(exp_walls)
        
        if sum_exp == 0:
            return -1e6, np.zeros_like(points_flat)
        
        # Weights
        w_pairs = exp_pairs / sum_exp
        w_walls = exp_walls / sum_exp
        
        soft_min_val = m - (1.0 / beta) * np.log(sum_exp)
        
        # Objective is -soft_min_val
        obj = -soft_min_val
        
        # Gradient
        grad = np.zeros_like(points_flat)
        
        # Pairwise gradient contribution
        # d/dx_i (d_ij) = (x_i - x_j) / d_ij
        # We have weights w_ij for each pair.
        # grad_i += sum_j w_ij * (x_i - x_j) / d_ij
        
        # Vectorized pairwise gradient
        # dists matrix D
        # w_pairs is same shape as D (upper triangle filled, lower 0? No, D is symmetric, w_pairs is symmetric if we compute on full matrix)
        # We computed exp_pairs on full matrix with diagonal inf -> exp(-inf)=0. So diagonal is 0.
        # w_pairs is symmetric.
        
        # grad_i = sum_j w_ij * (x_i - x_j) / d_ij
        # Note: d_ij is the distance between i and j.
        # Let's use broadcasting.
        # diff = x_i - x_j
        # term = w_ij * diff / d_ij
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        # d_ij is in dists
        # Avoid division by zero (diagonal is inf, w is 0, so 0/inf = 0)
        # But off-diagonal d_ij > 0.
        
        # w_pairs shape (N, N)
        # diff shape (N, N, 3)
        # dists shape (N, N)
        
        # term_k = w_pairs[i,j] * diff[i,j] / dists[i,j]
        # grad[i] = sum_j term_k
        
        # dists has inf on diagonal. w_pairs has 0 on diagonal.
        # 0 * inf is nan. We need to handle this.
        # Set dists diagonal to 1 for division, since weight is 0.
        safe_dists = dists.copy()
        np.fill_diagonal(safe_dists, 1.0)
        
        # term = w_pairs[:, :, np.newaxis] * diff / safe_dists[:, :, np.newaxis]
        term = w_pairs[:, :, np.newaxis] * diff / safe_dists[:, :, np.newaxis]
        
        # Sum over j (axis 1) to get grad for i
        grad_pairs = np.sum(term, axis=1)
        
        # Wall gradient contribution
        # d/dx_ik (0.5 - |x_ik|) = -sign(x_ik)
        # grad_walls_i_k = w_walls[i,k] * (-sign(x_ik))
        
        signs = np.sign(pts)
        # Handle 0: sign(0) is 0. Gradient of |x| at 0 is undefined, but subgradient is [-1, 1].
        # Using 0 is a valid choice for subgradient.
        grad_walls = -w_walls * signs
        
        # Combine
        grad = grad_pairs + grad_walls
        
        # Objective is -soft_min, so gradient is -grad_soft_min
        # Wait.
        # f(x) = -soft_min(x)
        # df/dx = - d(soft_min)/dx
        # My grad calculation above was for d(soft_min)/dx?
        # Let's check.
        # soft_min = m - 1/beta log(sum exp)
        # d(soft_min)/dx = -1/beta * 1/sum_exp * sum exp_k * (-beta) d_k' = sum exp_k d_k' / sum_exp = sum w_k d_k'
        # So the gradient I calculated (grad_pairs + grad_walls) is d(soft_min)/dx.
        # So df/dx = -(grad_pairs + grad_walls).
        
        grad = -(grad_pairs + grad_walls)
        
        return obj, grad

    def score(points):
        dists = np.linalg.norm(points[:, np.newaxis] - points[np.newaxis, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        pair_dists = dists / 2.0
        wall_dists = 0.5 - np.abs(points)
        return min(np.min(pair_dists), np.min(wall_dists))

    best_score = -1e9
    best_points = None
    
    # Initial configurations
    m = int(np.ceil(N ** (1.0 / 3.0)))
    ticks = (np.arange(m) + 0.5) / m
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    grid = grid[:N] * 1.0 - 0.5
    np.random.seed(42)
    jittered = grid + np.random.uniform(-0.01, 0.01, grid.shape)
    
    np.random.seed(123)
    random_pts = np.random.uniform(-0.4, 0.4, (N, 3))
    
    candidates = [jittered, random_pts]
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    for init in candidates:
        if time.time() > deadline - 30:
            break
            
        x0 = init.flatten()
        
        try:
            res = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True, bounds=bounds, 
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.success:
                pts = res.x.reshape(N, 3)
                actual_r = score(pts)
                
                if np.all(np.abs(pts) <= 0.5) and np.isfinite(actual_r):
                    if actual_r > best_score:
                        best_score = actual_r
                        best_points = pts.copy()
                        
        except Exception as e:
            pass
            
    if best_points is None:
        best_points = jittered
        
    best_points = np.clip(best_points, -0.5 + 1e-9, 0.5 - 1e-9)
    
    return best_points
# EVOLVE-BLOCK-END
