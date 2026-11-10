# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates.

    """
    start_time = time.time()
    deadline = 550.0  # Safety margin
    
    n = 16
    d = 2
    
    def get_dists(points_flat):
        points = points_flat.reshape(n, d)
        return pdist(points)
    
    def objective_and_grad(points_flat, beta):
        dists = get_dists(points_flat)
        
        # Soft min and max
        # Use log-sum-exp trick for stability
        # soft_min = -1/beta * log(sum(exp(-beta * d)))
        # soft_max = 1/beta * log(sum(exp(beta * d)))
        
        # To avoid overflow in exp, we shift by max/min of the exponents
        # For soft_min: exponents are -beta * d. Max exponent is -beta * min(d).
        # Let m_min = min(d). Then sum(exp(-beta*d)) = exp(-beta*m_min) * sum(exp(-beta*(d-m_min)))
        # log(sum(exp(-beta*d))) = -beta*m_min + log(sum(exp(-beta*(d-m_min))))
        
        m_min = np.min(dists)
        m_max = np.max(dists)
        
        # Calculate weights for gradient
        # w_k = exp(-beta * (d_k - m_min)) for min part
        # w_k = exp(beta * (d_k - m_max)) for max part
        
        exp_min_terms = np.exp(-beta * (dists - m_min))
        sum_min = np.sum(exp_min_terms)
        
        exp_max_terms = np.exp(beta * (dists - m_max))
        sum_max = np.sum(exp_max_terms)
        
        # Values
        log_soft_min = -m_min - (1.0/beta) * np.log(sum_min)
        log_soft_max = m_max + (1.0/beta) * np.log(sum_max)
        
        val = log_soft_max - log_soft_min
        # We want to minimize -val, so objective is val
        # Wait, maximize (dmin/dmax)^2 -> maximize log(dmin) - log(dmax).
        # So minimize -(log(dmin) - log(dmax)) = log(dmax) - log(dmin).
        # My val is log_soft_max - log_soft_min. So minimizing val is correct.
        
        # Gradient calculation
        # d(log_soft_min)/d(p_i) = -1/beta * (1/sum_min) * sum_k (d(exp_min_k)/d(p_i))
        # d(exp_min_k)/d(p_i) = exp_min_k * (-beta) * d(d_k)/d(p_i)
        # So d(log_soft_min)/d(p_i) = (1/sum_min) * sum_k (exp_min_k * d(d_k)/d(p_i))
        
        # d(log_soft_max)/d(p_i) = 1/beta * (1/sum_max) * sum_k (d(exp_max_k)/d(p_i))
        # d(exp_max_k)/d(p_i) = exp_max_k * beta * d(d_k)/d(p_i)
        # So d(log_soft_max)/d(p_i) = (1/sum_max) * sum_k (exp_max_k * d(d_k)/d(p_i))
        
        # Total Grad = d(log_soft_max)/d(p_i) - d(log_soft_min)/d(p_i)
        
        points = points_flat.reshape(n, d)
        grad = np.zeros_like(points)
        
        # Precompute pairwise differences
        # diff[i, j] = p_i - p_j
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        # dists matrix
        dists_matrix = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists_matrix, 1.0) # Avoid div by zero, but we mask it
        
        # We need to map the 120 distances from pdist back to the n x n matrix indices
        # pdist returns distances for i < j
        # We can iterate over i < j
        
        # Weights for min and max parts
        # Note: exp_min_terms and exp_max_terms are vectors of length 120
        # We need to map these to the (i, j) pairs
        
        # To be efficient, we can compute the gradient contribution for each pair
        # However, creating a full n x n weight matrix is easier.
        
        w_min_matrix = np.zeros((n, n))
        w_max_matrix = np.zeros((n, n))
        
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                w_min_matrix[i, j] = exp_min_terms[idx] / sum_min
                w_min_matrix[j, i] = exp_min_terms[idx] / sum_min
                w_max_matrix[i, j] = exp_max_terms[idx] / sum_max
                w_max_matrix[j, i] = exp_max_terms[idx] / sum_max
                idx += 1
        
        # Gradient of log_soft_min w.r.t p_i:
        # sum_j (w_min_matrix[i, j] * (p_i - p_j) / d_ij)
        # Note: d(d_ij)/d(p_i) = (p_i - p_j) / d_ij
        
        # Gradient of log_soft_max w.r.t p_i:
        # sum_j (w_max_matrix[i, j] * (p_i - p_j) / d_ij)
        
        # Total Grad_i = sum_j ( (w_max_matrix[i, j] - w_min_matrix[i, j]) * (p_i - p_j) / d_ij )
        
        # We can vectorize this using matrix multiplication if we construct the weight matrix properly.
        # Let W = w_max_matrix - w_min_matrix
        # Grad_i = sum_j W[i, j] * (p_i - p_j) / d_ij
        
        # This is not a standard matrix mult because of the 1/d_ij term.
        # But we can compute it.
        
        W = w_max_matrix - w_min_matrix
        
        # diff is (n, n, 2)
        # dists_matrix is (n, n)
        
        # Term for each i: sum_j W[i, j] * diff[i, j] / dists_matrix[i, j]
        
        # We can do:
        # weighted_diff = W[:, :, np.newaxis] * diff / dists_matrix[:, :, np.newaxis]
        # grad = np.sum(weighted_diff, axis=1)
        
        # Handle diagonal (dists_matrix[i,i]=1, diff[i,i]=0, W[i,i]=0) so it's safe.
        
        weighted_diff = W[:, :, np.newaxis] * diff / dists_matrix[:, :, np.newaxis]
        grad = np.sum(weighted_diff, axis=1)
        
        return val, grad.flatten()

    def optimize_from_seed(points):
        points_flat = points.flatten()
        
        # Annealing beta
        betas = [10, 50, 100, 500]
        
        best_val = np.inf
        best_points = points_flat
        
        for beta in betas:
            if time.time() > deadline:
                break
                
            # Run minimize
            # Use L-BFGS-B or BFGS
            # We provide the gradient
            res = minimize(
                objective_and_grad,
                points_flat,
                args=(beta,),
                jac=True, # We return (val, grad)
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < best_val:
                best_val = res.fun
                best_points = res.x
                
        return best_points, best_val

    # Generate seeds
    seeds = []
    
    # Seed 1: 4x4 Grid
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    X, Y = np.meshgrid(x, y)
    p_grid = np.column_stack((X.ravel(), Y.ravel()))
    p_grid -= p_grid.mean(axis=0)
    seeds.append(p_grid)
    
    # Seed 2: Random
    np.random.seed(123)
    p_rand = np.random.randn(n, d)
    p_rand -= p_rand.mean(axis=0)
    seeds.append(p_rand)
    
    # Seed 3: Two concentric squares
    # Inner square
    s1 = 0.5
    p_inner = np.array([
        [s1, s1], [s1, -s1], [-s1, -s1], [-s1, s1]
    ])
    # Outer square
    s2 = 1.5
    p_outer = np.array([
        [s2, s2], [s2, -s2], [-s2, -s2], [-s2, s2]
    ])
    p_conc = np.vstack([p_inner, p_outer])
    # Add 8 more points? 
    # Let's just use a distorted grid
    # Or a hexagonal like structure
    # For now, just grid and random.
    
    # Add a few random restarts
    for i in range(5):
        np.random.seed(100 + i)
        p_r = np.random.randn(n, d)
        p_r -= p_r.mean(axis=0)
        # Normalize to unit circle roughly
        max_d = np.max(pdist(p_r))
        if max_d > 0:
            p_r /= max_d
        seeds.append(p_r)
        
    best_overall_points = None
    best_overall_val = np.inf
    
    for seed in seeds:
        if time.time() > deadline:
            break
            
        try:
            res_points, res_val = optimize_from_seed(seed)
            if res_val < best_overall_val:
                best_overall_val = res_val
                best_overall_points = res_points
        except Exception as e:
            continue
            
    if best_overall_points is None:
        # Fallback to grid
        best_overall_points = seeds[0].flatten()
        
    final_points = best_overall_points.reshape(n, d)
    
    # Final polish: Ensure finite and shape correct
    if not np.all(np.isfinite(final_points)):
        final_points = seeds[0]
        
    return final_points

# EVOLVE-BLOCK-END
