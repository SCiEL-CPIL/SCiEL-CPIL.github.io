import numpy as np
import time

DIMENSIONS = [2, 2, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target Tensor
    # U index: i*m + j (size n*m)
    # V index: j*p + k (size m*p)
    # W index: k*n + i (size p*n)
    
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    
    T = np.zeros((U_dim, V_dim, W_dim))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    # Try to find rank 14 decomposition
    rank = 14
    best_score = float('inf')
    best_U, best_V, best_W = None, None, None
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    np.random.seed(42)
    
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        U = np.random.randn(U_dim, rank)
        V = np.random.randn(V_dim, rank)
        W = np.random.randn(W_dim, rank)
        
        # Normalize to help convergence
        # U /= np.linalg.norm(U, axis=0, keepdims=True)
        # V /= np.linalg.norm(V, axis=0, keepdims=True)
        # W /= np.linalg.norm(W, axis=0, keepdims=True)
        
        for it in range(1000):
            # Update U
            # T_u (V_dim x W_dim) = sum_r U[u, r] * (V[:, r] * W[:, r])
            # Reshape T to (U_dim, V_dim * W_dim)
            T_flat = T.reshape(U_dim, V_dim * W_dim)
            
            # Precompute M_r vectors
            # M_r is V[:, r] * W[:, r] reshaped to (V_dim * W_dim,)
            M = (V[:, :, None] * W[:, None, :]).reshape(V_dim * W_dim, rank).T # Shape (rank, V_dim * W_dim)
            
            # Solve T_flat[u] = M @ U[u]
            # U is (U_dim, rank)
            # We can solve all u at once if we stack M? No, M is same for all u.
            # T_flat (U_dim, V_dim*W_dim) = U (U_dim, rank) @ M.T (rank, V_dim*W_dim)
            # So U = T_flat @ M.T^+
            # Use lstsq
            try:
                U_new, residuals, rank_ls, s = np.linalg.lstsq(M.T, T_flat.T, rcond=None)
                U = U_new.T
            except:
                pass
                
            # Update V
            T_flat_v = T.reshape(V_dim, U_dim * W_dim)
            # M_r is U[:, r] * W[:, r]
            M_v = (U[:, :, None] * W[:, None, :]).reshape(U_dim * W_dim, rank).T
            try:
                V_new, _, _, _ = np.linalg.lstsq(M_v.T, T_flat_v.T, rcond=None)
                V = V_new.T
            except:
                pass
                
            # Update W
            T_flat_w = T.reshape(W_dim, U_dim * V_dim)
            M_w = (U[:, :, None] * V[:, None, :]).reshape(U_dim * V_dim, rank).T
            try:
                W_new, _, _, _ = np.linalg.lstsq(M_w.T, T_flat_w.T, rcond=None)
                W = W_new.T
            except:
                pass
                
            # Check error
            # Reconstruct
            # T_rec = sum_r U[:, r] * V[:, r] * W[:, r]
            # Efficient check:
            # Flatten T and T_rec
            T_rec_flat = np.einsum('ur,vr,wr->uvw', U, V, W).reshape(U_dim, V_dim * W_dim)
            err = np.linalg.norm(T_flat - T_rec_flat)
            
            if err < 1e-6:
                best_U, best_V, best_W = U, V, W
                best_score = err
                break
                
        if best_score < 1e-6:
            break
            
    if best_score < 1e-6:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Fallback to schoolbook
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
