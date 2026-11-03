import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin
    
    # Lower bound for rank is max(n*m, m*p, p*n)
    min_rank = max(n * m, m * p, p * n)
    
    # Schoolbook rank
    schoolbook_rank = n * m * p
    max_rank = schoolbook_rank
    
    # Precompute index arrays for vectorization
    # For U update:
    # U[i*m+j, :] = sum_k V[j*p+k, :] * W[k*n+i, :]
    # We iterate over all (i,j,k) combinations to build the indices for V and W
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    V_idx_for_U = j_idx * p + k_idx
    W_idx_for_U = k_idx * n + i_idx
    
    # For V update:
    # V[j*p+k, :] = sum_i U[i*m+j, :] * W[k*n+i, :]
    v_flat = np.arange(m * p)
    j_idx_v = v_flat // p
    k_idx_v = v_flat % p
    i_idx_v = np.repeat(np.arange(n), m * p)
    
    U_idx_for_V = i_idx_v * m + j_idx_v
    W_idx_for_V = k_idx_v * n + i_idx_v
    
    # For W update:
    # W[k*n+i, :] = sum_j U[i*m+j, :] * V[j*p+k, :]
    w_flat = np.arange(p * n)
    k_idx_w = w_flat // n
    i_idx_w = w_flat % n
    j_idx_w = np.repeat(np.arange(m), p * n)
    
    U_idx_for_W = i_idx_w * m + j_idx_w
    V_idx_for_W = j_idx_w * p + k_idx_w

    for R in range(min_rank, max_rank + 1):
        if time.time() > deadline - 10.0:
            break
        
        found_exact = False
        best_error = float('inf')
        best_U, best_V, best_W = None, None, None
        
        # Try more random initializations now that each is faster
        num_inits = 10 if R < 30 else 5
        
        for init_seed in range(num_inits):
            if time.time() > deadline - 10.0:
                break
            
            np.random.seed(init_seed + R * 1000)
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            # Normalize initial matrices to avoid scaling issues
            U /= np.linalg.norm(U, axis=0, keepdims=True)
            V /= np.linalg.norm(V, axis=0, keepdims=True)
            W /= np.linalg.norm(W, axis=0, keepdims=True)
            
            for it in range(100):
                if time.time() > deadline - 10.0:
                    break
                
                # Update U
                V_subset = V[V_idx_for_U, :] # Shape (n*m*p, R)
                W_subset = W[W_idx_for_U, :] # Shape (n*m*p, R)
                
                # Reshape to (n, m, p, R)
                V_sub_reshaped = V_subset.reshape(n, m, p, R)
                W_sub_reshaped = W_subset.reshape(n, m, p, R)
                
                A_u = np.sum(V_sub_reshaped * W_sub_reshaped, axis=2) # Shape (n, m, R)
                A_u = A_u.reshape(n * m, R)
                
                norms_sq = np.sum(A_u * A_u, axis=1)
                norms_sq[norms_sq == 0] = 1e-9
                U = A_u / norms_sq[:, np.newaxis]
                
                # Update V
                U_subset = U[U_idx_for_V, :] # Shape (n*m*p, R)
                W_subset_v = W[W_idx_for_V, :] # Shape (n*m*p, R)
                
                U_sub_reshaped = U_subset.reshape(n, m, p, R)
                W_sub_v_reshaped = W_subset_v.reshape(n, m, p, R)
                
                A_v = np.sum(U_sub_reshaped * W_sub_v_reshaped, axis=0) # Shape (m, p, R)
                A_v = A_v.reshape(m * p, R)
                
                norms_sq_v = np.sum(A_v * A_v, axis=1)
                norms_sq_v[norms_sq_v == 0] = 1e-9
                V = A_v / norms_sq_v[:, np.newaxis]
                
                # Update W
                U_subset_w = U[U_idx_for_W, :] # Shape (m*p*n, R)
                V_subset_w = V[V_idx_for_W, :] # Shape (m*p*n, R)
                
                U_sub_w_reshaped = U_subset_w.reshape(p, n, m, R)
                V_sub_w_reshaped = V_subset_w.reshape(p, n, m, R)
                
                A_w = np.sum(U_sub_w_reshaped * V_sub_w_reshaped, axis=2) # Shape (p, n, R)
                A_w = A_w.reshape(p * n, R)
                
                norms_sq_w = np.sum(A_w * A_w, axis=1)
                norms_sq_w[norms_sq_w == 0] = 1e-9
                W = A_w / norms_sq_w[:, np.newaxis]
                
                # Calculate error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        u_vec = U[i * m + j, :]
                        for k in range(p):
                            v_vec = V[j * p + k, :]
                            w_vec = W[k * n + i, :]
                            val = np.dot(u_vec, v_vec * w_vec)
                            err += (val - 1.0) ** 2
                
                if err < best_error:
                    best_error = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    
                if best_error < 1e-8:
                    break
            
            # Check if we found an exact solution
            if best_error < 1e-6:
                # Try rounding to half-integers
                U_int = np.round(best_U * 2) / 2
                V_int = np.round(best_V * 2) / 2
                W_int = np.round(best_W * 2) / 2
                
                is_exact = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_int[i * m + j, :], V_int[j * p + k, :] * W_int[k * n + i, :])
                            if abs(val - 1.0) > 1e-9:
                                is_exact = False
                                break
                        if not is_exact:
                            break
                    if not is_exact:
                        break
                
                if is_exact:
                    return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
        
        # If not exact at this rank, try next rank

    # Fallback to schoolbook if no low-rank solution found
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
