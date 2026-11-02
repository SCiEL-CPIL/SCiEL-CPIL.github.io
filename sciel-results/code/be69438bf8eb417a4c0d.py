import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Function to build the target tensor
    def get_target_tensor():
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # T[i*m+j, j*p+k, k*n+i] = 1
                    T[i * m + j, j * p + k, k * n + i] = 1
        return T

    T_target = get_target_tensor()
    
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    # Try to find a lower rank decomposition
    # We attempt rank 50
    target_rank = 50
    
    # Initialize with schoolbook, dropping last 10 columns
    U_init = U_sb[:, :target_rank].copy()
    V_init = V_sb[:, :target_rank].copy()
    W_init = W_sb[:, :target_rank].copy()
    
    # Flatten parameters for optimization
    params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    def error_func(params):
        U = params[:n * m * target_rank].reshape(n * m, target_rank)
        V = params[n * m * target_rank : n * m * target_rank + m * p * target_rank].reshape(m * p, target_rank)
        W = params[n * m * target_rank + m * p * target_rank :].reshape(p * n, target_rank)
        
        # Compute tensor decomposition
        # T_approx[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # This is a 3-way tensor contraction.
        # Efficient way:
        # For each r, add outer product of U[:,r], V[:,r], W[:,r]
        # But shapes are flattened.
        # U[:,r] is vector of length n*m.
        # We need to reshape U[:,r] to (n,m), V[:,r] to (m,p), W[:,r] to (p,n).
        
        T_approx = np.zeros((n, m, p, n)) # Temporary storage? No, target is (n*m, m*p, p*n)
        # Actually, we can just compute the error on the non-zero entries of T_target.
        # T_target has 1s at (i*m+j, j*p+k, k*n+i).
        
        err = 0.0
        for r in range(target_rank):
            u_vec = U[:, r]
            v_vec = V[:, r]
            w_vec = W[:, r]
            
            # Reshape to tensor components
            u_mat = u_vec.reshape(n, m)
            v_mat = v_vec.reshape(m, p)
            w_mat = w_vec.reshape(p, n)
            
            # The term contributed by this rank-1 component to T[i*m+j, j*p+k, k*n+i]
            # is u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
                        # Compare with T_target[i*m+j, j*p+k, k*n+i] which is 1
                        err += (val - 1) ** 2
                        
        return err

    # Use Nelder-Mead or BFGS
    # Nelder-Mead is derivative-free and might be more robust for this small problem
    result = minimize(error_func, params, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8})
    
    if result.fun < 1e-6:
        # Extract and round
        U_opt = result.x[:n * m * target_rank].reshape(n * m, target_rank)
        V_opt = result.x[n * m * target_rank : n * m * target_rank + m * p * target_rank].reshape(m * p, target_rank)
        W_opt = result.x[n * m * target_rank + m * p * target_rank :].reshape(p * n, target_rank)
        
        # Round to nearest 0.5
        U_opt = np.round(U_opt * 2) / 2
        V_opt = np.round(V_opt * 2) / 2
        W_opt = np.round(W_opt * 2) / 2
        
        # Verify exactness
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    sum_val = 0.0
                    for r in range(target_rank):
                        sum_val += U_opt[i * m + j, r] * V_opt[j * p + k, r] * W_opt[k * n + i, r]
                    if abs(sum_val - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid: break
            if not valid: break
            
        if valid:
            return {
                "U": U_opt.tolist(),
                "V": V_opt.tolist(),
                "W": W_opt.tolist()
            }
            
    # Fallback to schoolbook
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
# EVOLVE-BLOCK-END
