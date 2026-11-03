import numpy as np
import time
import random

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Target tensor T
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1
                
    # Flatten T for linear algebra
    T_flat = T.flatten()
    
    # Function to check if a decomposition is exact
    def check_decomposition(U, V, W):
        # U: (dim_u, r)
        # V: (dim_v, r)
        # W: (dim_w, r)
        # Reconstruct T
        T_recon = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
        r = U.shape[1]
        for l in range(r):
            u_col = U[:, l]
            v_col = V[:, l]
            w_col = W[:, l]
            T_recon += np.einsum('i,j,k->ijk', u_col, v_col, w_col)
            
        # Check if close to T
        max_diff = np.max(np.abs(T_recon - T))
        return max_diff < 1e-9

    # Initialize with schoolbook (Rank 60)
    rank_schoolbook = n * m * p
    U_school = np.zeros((dim_u, rank_schoolbook), dtype=np.int64)
    V_school = np.zeros((dim_v, rank_schoolbook), dtype=np.int64)
    W_school = np.zeros((dim_w, rank_schoolbook), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_schoolbook
    
    # Time management
    start_time = time.time()
    deadline = start_time + 110 # 10s margin for 120s limit
    
    # Search for lower rank
    target_ranks = [55, 50, 45]
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        max_attempts = 5 
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
                
            # Initialize U, V with small random integers
            U = np.random.randint(-2, 3, size=(dim_u, target_rank))
            V = np.random.randint(-2, 3, size=(dim_v, target_rank))
            
            # Build M matrix for solving W
            M = np.zeros((dim_u * dim_v * dim_w, dim_w * target_rank), dtype=np.float64)
            
            for i in range(dim_u):
                for j in range(dim_v):
                    row_base = (i * dim_v + j) * dim_w
                    for k in range(dim_w):
                        row_idx = row_base + k
                        M[row_idx, k*target_rank : (k+1)*target_rank] = U[i, :] * V[j, :]
                        
            # Solve M * W_flat = T_flat
            try:
                W_flat, residuals, rank_M, s = np.linalg.lstsq(M, T_flat, rcond=None)
            except:
                continue
                
            W = W_flat.reshape(dim_w, target_rank)
            
            # Check if the solution is exact
            if check_decomposition(U, V, W):
                W_int = np.round(W).astype(np.int64)
                if check_decomposition(U, V, W_int):
                    best_U, best_V, best_W = U, V, W_int
                    best_rank = target_rank
                    break
                else:
                    W_half = np.round(W * 2) / 2.0
                    if check_decomposition(U, V, W_half):
                        best_U, best_V, best_W = U, V, W_half
                        best_rank = target_rank
                        break
            else:
                W_int = np.round(W).astype(np.int64)
                if check_decomposition(U, V, W_int):
                    best_U, best_V, best_W = U, V, W_int
                    best_rank = target_rank
                    break
                W_half = np.round(W * 2) / 2.0
                if check_decomposition(U, V, W_half):
                    best_U, best_V, best_W = U, V, W_half
                    best_rank = target_rank
                    break

    # Convert to lists for return
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
