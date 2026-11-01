import numpy as np
import scipy.optimize
import time

N = 14
D = 3

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Parameters for soft-min/max
    k = 20.0
    
    def get_distances(X):
        # X is (N, D)
        # Returns (N, N) distance matrix
        sq_dists = np.sum(X**2, axis=1)[:, None] + np.sum(X**2, axis=1)[None, :] - 2 * X @ X.T
        np.fill_diagonal(sq_dists, 0)
        sq_dists = np.maximum(sq_dists, 0)
        dists = np.sqrt(sq_dists)
        return dists

    def objective_and_grad(flat_X):
        X = flat_X.reshape(N, D)
        dists = get_distances(X)
        
        # Mask for i < j
        i, j = np.triu_indices(N, k=1)
        d = dists[i, j]
        
        # Avoid division by zero
        d_safe = np.maximum(d, 1e-8)
        
        # Soft Max
        exp_max = np.exp(k * d_safe)
        Z_max = np.sum(exp_max)
        S_max = (1.0 / k) * np.log(Z_max)
        
        # Soft Min
        exp_min = np.exp(-k * d_safe)
        Z_min = np.sum(exp_min)
        S_min = -(1.0 / k) * np.log(Z_min)
        
        ratio = S_max / S_min
        val = ratio * ratio
        
        # Gradient
        # dS_max/dX and dS_min/dX
        # dS_max/dx_i = (1/Z_max) * sum_j exp(k*d_ij) * (x_i - x_j)/d_ij
        # dS_min/dx_i = (1/Z_min) * sum_j exp(-k*d_ij) * (x_i - x_j)/d_ij
        
        grad_S_max = np.zeros_like(X)
        grad_S_min = np.zeros_like(X)
        
        # We can compute this efficiently
        # For each pair (i, j), contribution to grad_i and grad_j
        
        # Vectorized approach:
        # W_max[i,j] = exp(k*d_ij) / Z_max
        # W_min[i,j] = exp(-k*d_ij) / Z_min
        
        W_max = exp_max / Z_max
        W_min = exp_min / Z_min
        
        # Only upper triangle matters for the sum, but the gradient formula sums over all j != i
        # Since d_ij = d_ji, we can use the full matrix but need to be careful with weights.
        # The formula derived was sum over j != i.
        
        # Let's construct the full weight matrix
        W_max_full = np.zeros((N, N))
        W_max_full[i, j] = W_max
        W_max_full[j, i] = W_max
        
        W_min_full = np.zeros((N, N))
        W_min_full[i, j] = W_min
        W_min_full[j, i] = W_min
        
        # Direction vectors
        # diff[i,j] = (x_i - x_j) / d_ij
        diff = np.zeros((N, N, D))
        for a in range(N):
            for b in range(N):
                if a != b:
                    d_ab = d_safe[i, j] if i[a] == a and j[a] == b else d_safe[i, j] # This indexing is messy
                    # Better to compute diff directly from X
                    pass
        
        # Let's re-calculate diff properly
        # dists is (N, N)
        # diff[i,j] = (X[i] - X[j]) / dists[i,j]
        # We can do this with broadcasting
        X_i = X[:, None, :]
        X_j = X[None, :, :]
        diff_vec = X_i - X_j
        # dists is (N, N), need to expand to (N, N, 1)
        dists_exp = dists[:, :, None]
        # Avoid div by zero
        dists_safe_exp = np.maximum(dists_exp, 1e-8)
        unit_vec = diff_vec / dists_safe_exp
        
        # grad_S_max[i] = sum_j W_max_full[i,j] * unit_vec[i,j]
        grad_S_max = np.sum(W_max_full[:, :, None] * unit_vec, axis=1)
        
        # grad_S_min[i] = sum_j W_min_full[i,j] * unit_vec[i,j]
        grad_S_min = np.sum(W_min_full[:, :, None] * unit_vec, axis=1)
        
        # d(val)/dX = 2 * S_max * S_min^-2 * grad_S_max - 2 * S_max^2 * S_min^-3 * grad_S_min
        factor1 = 2 * S_max / (S_min * S_min)
        factor2 = -2 * (S_max * S_max) / (S_min * S_min * S_min)
        
        grad = (factor1 * grad_S_max + factor2 * grad_S_min).flatten()
        
        return val, grad

    best_score = np.inf
    best_X = None
    
    # Try a few different random initializations
    rng = np.random.default_rng(42)
    
    num_restarts = 60
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Random start
        X0 = rng.uniform(-1, 1, size=(N, D))
        
        # Optional: Center and scale initial guess to be in a reasonable range
        X0 = X0 - np.mean(X0, axis=0)
        max_d = np.max(get_distances(X0))
        if max_d > 0:
            X0 = X0 / max_d # Scale so max dist is 1
            
        try:
            res = scipy.optimize.minimize(
                objective_and_grad, 
                X0.flatten(), 
                method='L-BFGS-B', 
                jac=True, # Use analytic gradient
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            X_opt = res.x.reshape(N, D)
            
            # Calculate true score
            d_true = get_distances(X_opt)
            i, j = np.triu_indices(N, k=1)
            d_pairs = d_true[i, j]
            d_min = np.min(d_pairs)
            d_max = np.max(d_pairs)
            
            if d_min < 1e-6:
                score = np.inf
            else:
                score = (d_max / d_min) ** 2
                
            if score < best_score:
                best_score = score
                best_X = X_opt.copy()
                
        except Exception:
            pass

    if best_X is None:
        # Fallback to grid if optimization failed completely
        side = int(np.ceil(N ** (1 / D)))
        import itertools
        grid = list(itertools.product(range(side), repeat=D))[:N]
        best_X = np.array(grid, dtype=float)
        
    return best_X
