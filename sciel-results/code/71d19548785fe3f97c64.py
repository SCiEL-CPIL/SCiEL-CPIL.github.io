import numpy as np
import scipy.optimize as opt

def construct():
    n, m, p = 4, 4, 5
    
    # Non-zero indices of the tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                indices.append((u_idx, v_idx, w_idx))
                
    # Target values are all 1
    target_vals = np.ones(len(indices))
    
    def get_residuals(params, rank):
        # params: flattened U, V, W
        size_u = n * m * rank
        size_v = m * p * rank
        size_w = p * n * rank
        
        U = params[:size_u].reshape(n*m, rank)
        V = params[size_u:size_u+size_v].reshape(m*p, rank)
        W = params[size_u+size_v:size_u+size_v+size_w].reshape(p*n, rank)
        
        residuals = np.zeros(len(indices))
        for idx, (u_i, v_i, w_i) in enumerate(indices):
            val = np.dot(U[u_i, :], np.dot(V[v_i, :], W[w_i, :]))
            # val = sum_r U[u_i, r] V[v_i, r] W[w_i, r]
            # This is a scalar.
            residuals[idx] = val - target_vals[idx]
        return residuals

    def get_grad(params, rank):
        size_u = n * m * rank
        size_v = m * p * rank
        size_w = p * n * rank
        
        U = params[:size_u].reshape(n*m, rank)
        V = params[size_u:size_u+size_v].reshape(m*p, rank)
        W = params[size_u+size_v:size_u+size_v+size_w].reshape(p*n, rank)
        
        # Compute predictions
        preds = np.zeros(len(indices))
        for idx, (u_i, v_i, w_i) in enumerate(indices):
            preds[idx] = np.dot(U[u_i, :], np.dot(V[v_i, :], W[w_i, :]))
            
        errors = preds - target_vals # residuals = val - 1, so error = val - 1
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for idx, (u_i, v_i, w_i) in enumerate(indices):
            err = errors[idx]
            # d L / d U[u_i, r] = 2 * err * V[v_i, r] * W[w_i, r]
            # We minimize sum(residuals^2), residual = val - 1.
            # d L / d val = 2 * (val - 1)
            # d val / d U[u_i, r] = V[v_i, r] * W[w_i, r]
            factor = 2 * err
            grad_U[u_i, :] += factor * (V[v_i, :] * W[w_i, :])
            grad_V[v_i, :] += factor * (U[u_i, :] * W[w_i, :])
            grad_W[w_i, :] += factor * (U[u_i, :] * V[v_i, :])
            
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

    def try_rank(rank, num_restarts=5):
        size_u = n * m * rank
        size_v = m * p * rank
        size_w = p * n * rank
        total_size = size_u + size_v + size_w
        
        np.random.seed(42)
        
        for i in range(num_restarts):
            # Random initialization
            # Scale to avoid overflow/underflow
            init_params = np.random.randn(total_size) * 0.5
            
            # Optimization
            try:
                res = opt.minimize(
                    get_residuals, 
                    init_params, 
                    args=(rank,), 
                    method='L-BFGS-B', 
                    jac=get_grad,
                    options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8}
                )
            except Exception:
                continue
                
            if res.success or np.linalg.norm(res.fun) < 1e-4:
                # Check if it's close enough to integer/half-integer
                # Round to nearest 0.5
                rounded_params = np.round(res.x * 2) / 2.0
                # Verify
                if np.linalg.norm(get_residuals(rounded_params, rank)) < 1e-9:
                    U = rounded_params[:size_u].reshape(n*m, rank)
                    V = rounded_params[size_u:size_u+size_v].reshape(m*p, rank)
                    W = rounded_params[size_u+size_v:size_u+size_v+size_w].reshape(p*n, rank)
                    return U, V, W
        return None

    # Try rank 64
    sol = try_rank(64, num_restarts=10)
    if sol:
        return {"U": sol[0].tolist(), "V": sol[1].tolist(), "W": sol[2].tolist()}
        
    # Fallback: Rank 80 Schoolbook
    # This ensures a valid output
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i*m + j
                v_idx = j*p + k
                w_idx = k*n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
