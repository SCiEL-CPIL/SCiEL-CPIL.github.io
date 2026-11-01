"""Tensor decomposition for 2x2 * 2x13 matrix multiplication."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin
    
    n, m, p = 2, 2, 13
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+k, k*p+j, j*n+i] = 1
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                T[i*m+k, k*p+j, j*n+i] = 1.0
    
    # Start with Strassen-based decomposition (rank 46)
    # Then try to reduce rank via optimization
    
    # First, build the rank-46 Strassen decomposition
    def build_strassen_decomp():
        U_local = np.array([
            [1, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 0, 1,  0, 1],
            [0, 1, 0, 0, 0,  1, 0],
            [1, 1, 0, 1, 0,  0,-1],
        ], dtype=float)
        
        V_local = np.array([
            [1, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1,-1, 0, 1, 0],
            [1, 0,-1, 0, 1, 0, 1],
        ], dtype=float)
        
        W_local = np.array([
            [1, 0, 0, 1, 0, 0, 1],
            [0,-1, 1, 0, 0, 1, 0],
            [0, 1,-1, 1,-1, 0, 0],
            [1, 1, 0,-1, 0,-1, 0],
        ], dtype=float)
        
        total_rank = 6 * 7 + 4
        U_full = np.zeros((n * m, total_rank), dtype=float)
        V_full = np.zeros((m * p, total_rank), dtype=float)
        W_full = np.zeros((p * n, total_rank), dtype=float)
        
        col_offset = 0
        for block in range(6):
            j0 = block * 2
            j1 = block * 2 + 1
            
            for t in range(7):
                for i in range(2):
                    for k in range(2):
                        U_full[i*m+k, col_offset + t] = U_local[i*2+k, t]
                
                for k in range(2):
                    for jl in range(2):
                        j_global = j0 + jl
                        V_full[k*p + j_global, col_offset + t] = V_local[k*2+jl, t]
                
                for jl in range(2):
                    for i in range(2):
                        j_global = j0 + jl
                        W_full[j_global*n + i, col_offset + t] = W_local[jl*2+i, t]
            
            col_offset += 7
        
        j = 12
        for i in range(n):
            for k in range(m):
                U_full[i*m+k, col_offset] = 1
                V_full[k*p+j, col_offset] = 1
                W_full[j*n+i, col_offset] = 1
                col_offset += 1
        
        return U_full, V_full, W_full
    
    U0, V0, W0 = build_strassen_decomp()
    best_U, best_V, best_W = U0, V0, W0
    best_rank = 46
    
    # Try to reduce rank by setting some columns to zero and re-optimizing
    # We'll try ranks 45, 44, ... down to some minimum
    
    def tensor_error(U, V, W):
        """Compute the error in the tensor decomposition."""
        err = 0.0
        for t in range(U.shape[1]):
            err += np.sum((np.einsum('a,b,c->abc', U[:,t], V[:,t], W[:,t]) - T) ** 2)
        return np.sqrt(err)
    
    def try_reduce_rank(rank_target, U_init, V_init, W_init, max_iter=50):
        """Try to find a valid decomposition with the given rank."""
        if time.time() > deadline - 10:
            return None
        
        # Initialize by taking the first rank_target columns from the current best
        U = U_init[:, :rank_target].copy()
        V = V_init[:, :rank_target].copy()
        W = W_init[:, :rank_target].copy()
        
        # Flatten the parameters
        def flatten_params(U, V, W):
            return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def unflatten_params(params):
            u_size = U.shape[0] * U.shape[1]
            v_size = V.shape[0] * V.shape[1]
            w_size = W.shape[0] * W.shape[1]
            U_flat = params[:u_size]
            V_flat = params[u_size:u_size+v_size]
            W_flat = params[u_size+v_size:u_size+v_size+w_size]
            return U_flat.reshape(U.shape), V_flat.reshape(V.shape), W_flat.reshape(W.shape)
        
        def objective(params):
            U, V, W = unflatten_params(params)
            return tensor_error(U, V, W)
        
        # Use L-BFGS-B for optimization
        x0 = flatten_params(U, V, W)
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', 
                            options={'maxiter': max_iter, 'maxfun': 500})
            
            if result.success or result.fun < 1e-6:
                U_opt, V_opt, W_opt = unflatten_params(result.x)
                if tensor_error(U_opt, V_opt, W_opt) < 1e-6:
                    return U_opt, V_opt, W_opt
        except Exception:
            pass
        
        return None
    
    # Try to reduce rank from 46 down
    for rank_try in range(45, 39, -1):
        if time.time() > deadline - 20:
            break
        
        result = try_reduce_rank(rank_try, best_U, best_V, best_W, max_iter=30)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = rank_try
            print(f"Reduced rank to {best_rank}")
    
    # Final verification
    final_err = tensor_error(best_U, best_V, best_W)
    if final_err > 1e-5:
        # Fall back to the Strassen decomposition
        best_U, best_V, best_W = U0, V0, W0
        best_rank = 46
    
    # Convert to lists with half-integer rounding
    def round_to_half(x):
        return round(x * 2) / 2
    
    U_list = [[round_to_half(v) for v in row] for row in best_U.tolist()]
    V_list = [[round_to_half(v) for v in row] for row in best_V.tolist()]
    W_list = [[round_to_half(v) for v in row] for row in best_W.tolist()]
    
    # Verify after rounding
    U_check = np.array(U_list)
    V_check = np.array(V_list)
    W_check = np.array(W_list)
    check_err = tensor_error(U_check, V_check, W_check)
    
    if check_err > 1e-4:
        # Use the unrounded Strassen decomposition
        U_list = [[round_to_half(v) for v in row] for row in U0.tolist()]
        V_list = [[round_to_half(v) for v in row] for row in V0.tolist()]
        W_list = [[round_to_half(v) for v in row] for row in W0.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
