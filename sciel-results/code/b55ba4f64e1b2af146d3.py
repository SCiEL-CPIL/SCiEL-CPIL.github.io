import numpy as np
from scipy.optimize import minimize
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Initial guess: 5x5x4 grid
    # 5 in x, 5 in y, 4 in z
    nx, ny, nz = 5, 5, 4
    # Ensure nx*ny*nz >= N
    # 5*5*4 = 100 exactly.
    
    # Generate grid
    x = np.linspace(-0.5, 0.5, nx)
    y = np.linspace(-0.5, 0.5, ny)
    z = np.linspace(-0.5, 0.5, nz)
    
    # Create meshgrid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    # If grid size > N, take first N. If < N, we need to adjust.
    # Here 5*5*4 = 100.
    c0 = grid[:N]
    
    # Add small jitter to break symmetry
    c0 = c0 + 0.01 * np.random.randn(N, 3)
    # Clip to be safely inside
    c0 = np.clip(c0, -0.49, 0.49)
    
    best_c = c0
    best_score = -np.inf
    
    # Objective function: Minimize negative soft-min radius
    beta = 50.0  # Sharpness of soft-min
    
    def objective_and_grad(c_flat):
        c = c_flat.reshape(N, 3)
        
        # Compute pair distances
        # Use broadcasting for all pairs
        # diff[i,j] = c[i] - c[j]
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        # Avoid division by zero
        dists = np.sqrt(dists_sq + 1e-12)
        
        # Pair radii: r_ij = dist_ij / 2
        r_pairs = dists / 2.0
        
        # Wall distances
        # r_wall_ik = 0.5 - abs(c_ik)
        r_walls = 0.5 - np.abs(c)
        
        # Combine all radii into a single vector
        # Number of pairs: N*(N-1)/2
        # Number of wall distances: N*3
        # We need to handle the upper triangle of the distance matrix for pairs to avoid double counting and self-interaction
        i_idx, j_idx = np.triu_indices(N, k=1)
        r_pair_vals = r_pairs[i_idx, j_idx]
        
        r_wall_vals = r_walls.ravel()
        
        all_r = np.concatenate([r_pair_vals, r_wall_vals])
        
        # Soft-min
        # To avoid overflow in exp, shift by max
        max_r = np.max(all_r)
        exp_terms = np.exp(-beta * (all_r - max_r))
        sum_exp = np.sum(exp_terms)
        weights = exp_terms / sum_exp
        
        soft_min_val = max_r - (1.0/beta) * np.log(sum_exp)
        
        obj_val = -soft_min_val
        
        # Gradient
        # d(obj)/dc = - d(soft_min)/dc
        # d(soft_min)/d(all_r_k) = weights[k]
        # So we need to map weights back to c
        
        grad = np.zeros_like(c)
        
        # Pair contributions
        # weight for pair (i,j) is weights[k]
        # d(r_pair_ij)/dc_i = 0.5 * (c_i - c_j) / dist_ij
        # d(r_pair_ij)/dc_j = 0.5 * (c_j - c_i) / dist_ij
        
        # We have weights for the upper triangle part of all_r
        # Let's separate them
        w_pairs = weights[:len(r_pair_vals)]
        w_walls = weights[len(r_pair_vals):]
        
        # Accumulate gradients for pairs
        # For each pair (i, j) with weight w:
        # grad[i] += w * 0.5 * (c[i] - c[j]) / dist[i,j]
        # grad[j] += w * 0.5 * (c[j] - c[i]) / dist[i,j]
        
        # Vectorized update is tricky with scatter_add. 
        # Given N=100, N^2=10000 pairs. Loop is 4950 iterations.
        # Python loop might be slow if done in pure python, but 4950 is small.
        # However, we call this many times.
        # Let's try to vectorize or use sparse matrix trick if possible, or just loop.
        # 4950 iterations per gradient eval.
        # L-BFGS-B might call grad 100-1000 times.
        # 500,000 iterations total. Should be fast in Python? Maybe.
        
        # Let's optimize the loop:
        for k in range(len(i_idx)):
            i = i_idx[k]
            j = j_idx[k]
            w = w_pairs[k]
            if w < 1e-10: continue
            
            d_vec = c[i] - c[j]
            d_norm = dists[i, j]
            
            if d_norm < 1e-10:
                continue
                
            factor = 0.5 * w / d_norm
            
            grad[i] += factor * d_vec
            grad[j] -= factor * d_vec
            
        # Wall contributions
        # r_wall_ik = 0.5 - |c_ik|
        # d(r_wall_ik)/dc_ik = -sign(c_ik)
        # weight is w_walls[m] corresponding to c[i, k]
        
        # Reshape w_walls to (N, 3)
        w_walls_grid = w_walls.reshape(N, 3)
        
        signs = np.sign(c)
        # If c is 0, sign is 0. This is fine, gradient is 0 there (subgradient).
        grad -= w_walls_grid * signs
        
        return obj_val, grad

    # Run optimization
    # We can try to run for a while and check time.
    
    try:
        res = minimize(
            objective_and_grad, 
            c0.ravel(), 
            method='L-BFGS-B', 
            jac=True, 
            bounds=[(-0.5, 0.5)] * (N*3),
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
        
        if time.time() < deadline:
            best_c = res.x.reshape(N, 3)
            
    except Exception as e:
        pass

    # Final Polish: Local search with smaller step or more iterations?
    # Or just return the best found.
    
    # Ensure validity
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
