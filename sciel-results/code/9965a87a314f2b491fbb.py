import numpy as np
import time
import random

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = np.zeros((dim_u, rank_school))
    V_school = np.zeros((dim_v, rank_school))
    W_school = np.zeros((dim_w, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    # Target tensor
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
                
    # Try to find a lower rank decomposition
    # We attempt rank r = rank_school - 1
    best_rank = rank_school
    best_decomp = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    start_time = time.time()
    deadline = 110  # 10s margin
    
    # Try ranks from rank_school - 1 down to rank_school - 5
    for target_rank in range(rank_school - 1, rank_school - 6, -1):
        if time.time() > deadline:
            break
            
        # Try a few random initializations for this rank
        for attempt in range(20):
            if time.time() > deadline:
                break
                
            # Initialize U, V, W with random sparse vectors
            # Use a mix of 0, 1, -1, 0.5, -0.5
            U = np.zeros((dim_u, target_rank))
            V = np.zeros((dim_v, target_rank))
            W = np.zeros((dim_w, target_rank))
            
            for r in range(target_rank):
                # Randomly select a few non-zero entries
                nz_u = np.random.choice(dim_u, size=min(3, dim_u), replace=False)
                nz_v = np.random.choice(dim_v, size=min(3, dim_v), replace=False)
                nz_w = np.random.choice(dim_w, size=min(3, dim_w), replace=False)
                
                for idx in nz_u:
                    U[idx, r] = random.choice([0.5, 1, -0.5, -1])
                for idx in nz_v:
                    V[idx, r] = random.choice([0.5, 1, -0.5, -1])
                for idx in nz_w:
                    W[idx, r] = random.choice([0.5, 1, -0.5, -1])
            
            # ALS Optimization
            for iteration in range(50):
                if time.time() > deadline:
                    break
                    
                # Update U
                for i in range(dim_u):
                    # T_i is a matrix of size (dim_v, dim_w)
                    # T_i[j, k] = T[i, j, k]
                    # We want U[i, r] such that sum_r U[i, r] V[r, j] W[r, k] = T[i, j, k]
                    # This is equivalent to:
                    # T_i = U_i @ (V * W^T) ? No.
                    # T[i, :, :] = sum_r U[i, r] outer(V[:, r], W[:, r])
                    # Let M_r = outer(V[:, r], W[:, r])
                    # T[i, :, :] = sum_r U[i, r] M_r
                    # This is a linear regression: T_i (flattened) = U_i @ (M_r flattened)
                    # M_flattened is a matrix of size (target_rank, dim_v * dim_w)
                    
                    # Construct M_matrix: (target_rank, dim_v * dim_w)
                    # M_matrix[r, j*dim_w + k] = V[r, j] * W[r, k]
                    M = np.zeros((target_rank, dim_v * dim_w))
                    for r in range(target_rank):
                        M[r, :] = np.outer(V[:, r], W[:, r]).flatten()
                    
                    T_i_flat = T[i, :, :].flatten()
                    # Solve U[i, :] = (M^T M)^-1 M^T T_i_flat
                    # Use lstsq for stability
                    try:
                        U[i, :], residuals, rank, sv = np.linalg.lstsq(M.T, T_i_flat, rcond=None)
                    except:
                        continue
                        
                # Update V
                for j in range(dim_v):
                    M = np.zeros((target_rank, dim_u * dim_w))
                    for r in range(target_rank):
                        M[r, :] = np.outer(U[:, r], W[:, r]).flatten()
                    T_j_flat = T[:, j, :].flatten()
                    try:
                        V[j, :], residuals, rank, sv = np.linalg.lstsq(M.T, T_j_flat, rcond=None)
                    except:
                        continue
                        
                # Update W
                for k in range(dim_w):
                    M = np.zeros((target_rank, dim_u * dim_v))
                    for r in range(target_rank):
                        M[r, :] = np.outer(U[:, r], V[:, r]).flatten()
                    T_k_flat = T[:, :, k].flatten()
                    try:
                        W[k, :], residuals, rank, sv = np.linalg.lstsq(M.T, T_k_flat, rcond=None)
                    except:
                        continue
                        
            # Round to nearest half-integer
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Verify
            # Reconstruct tensor
            T_recon = np.zeros((dim_u, dim_v, dim_w))
            for r in range(target_rank):
                T_recon += np.outer(np.outer(U_rounded[:, r], V_rounded[:, r]), W_rounded[:, r])
                
            if np.allclose(T_recon, T, atol=1e-9):
                best_rank = target_rank
                best_decomp = {
                    "U": U_rounded.tolist(),
                    "V": V_rounded.tolist(),
                    "W": W_rounded.tolist()
                }
                break # Found valid decomposition for this rank, try lower rank
                
    return best_decomp
