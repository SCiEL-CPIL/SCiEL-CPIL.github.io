import numpy as np
import time

def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin before the 150s kill
    
    n, m, p = 3, 4, 4
    Nm, Mp, Pn = n*m, m*p, p*n  # 12, 16, 12
    
    # Build tensor - only the 48 nonzero entries
    T_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_indices.append((i*m+j, j*p+k, k*n+i))
    T = np.zeros((Nm, Mp, Pn))
    for a, b, c in T_indices:
        T[a, b, c] = 1.0
    
    def verify_fast(U, V, W):
        R = U.shape[1]
        if R == 0:
            return False
        # T_rec[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        UV = U[:, None, :] * V[None, :, :]  # (Nm, Mp, R)
        UV_flat = UV.reshape(-1, R)  # (Nm*Mp, R)
        T_rec_flat = UV_flat @ W.T  # (Nm*Mp, Pn)
        T_rec = T_rec_flat.reshape(Nm, Mp, Pn)
        return np.allclose(T_rec, T, atol=1e-8)
    
    # Direct decomposition (rank 48)
    def make_direct():
        R = n * m * p
        U = np.zeros((Nm, R))
        V = np.zeros((Mp, R))
        W = np.zeros((Pn, R))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, idx] = 1.0
                    V[j*p+k, idx] = 1.0
                    W[k*n+i, idx] = 1.0
                    idx += 1
        return U, V, W
    
    # Block decomposition (rank 12) - derived from analysis
    def make_block_decomp():
        rank = 12
        U = np.zeros((Nm, rank))
        V = np.zeros((Mp, rank))
        W = np.zeros((Pn, rank))
        
        # For each j in 0..3, we use 3 ranks (one for each i in 0..2)
        for j in range(m):
            for i in range(n):
                r_idx = j * n + i
                a_idx = i * m + j
                b_idx = j * p + j # Since k=j
                c_idx = j * n + i # Since k=j, c = k*n + i = j*n + i
                
                U[a_idx, r_idx] = 1.0
                V[b_idx, r_idx] = 1.0
                W[c_idx, r_idx] = 1.0
                
        return U, V, W
    
    U_blk, V_blk, W_blk = make_block_decomp()
    
    if verify_fast(U_blk, V_blk, W_blk):
        best_U, best_V, best_W = U_blk, V_blk, W_blk
        best_rank = 12
    else:
        # Fallback to direct if verification fails (should not happen)
        U_dir, V_dir, W_dir = make_direct()
        best_U, best_V, best_W = U_dir, V_dir, W_dir
        best_rank = 48
        
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
