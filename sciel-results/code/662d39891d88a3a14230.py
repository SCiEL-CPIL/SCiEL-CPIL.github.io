import numpy as np
import time

def construct():
    n, m, p = 2, 3, 10
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    dim_u = n * m  # 6
    dim_v = m * p  # 30
    dim_w = p * n  # 20
    
    # Build target tensor T[u, v, w] = 1 for valid (i,j,k) combinations
    # u = i*m+j, v = j*p+k, w = k*n+i
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    def verify(U, V, W):
        """Check if U,V,W exactly decompose T (with half-integer rounding)."""
        U_r = np.round(U * 2) / 2.0
        V_r = np.round(V * 2) / 2.0
        W_r = np.round(W * 2) / 2.0
        T_rec = np.einsum('ur,vr,wr->uvw', U_r, V_r, W_r)
        return np.allclose(T_rec, T, atol=1e-9), U_r, V_r, W_r
    
    def get_strassen_decomposition():
        """Current rank-36 Strassen-based decomposition."""
        rank = 36
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        
        strassen_tensors = [
            ([1, 0, 1, 0], [1, 0, 0, 1], [1, 1, 1, -1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
            ([1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]),
            ([0, 0, 0, 1], [0, 0, 0, 1], [1, 0, 0, 1]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, -1, 0]),
            ([0, 1, -1, 0], [0, 0, 1, 0], [0, 0, 0, 1]),
            ([0, 0, 1, -1], [1, 0, 0, 0], [1, 0, 0, 0]),
        ]
        
        for block in range(5):
            k_offset = block * 2
            for t_idx, (u, v, w) in enumerate(strassen_tensors):
                col = block * 7 + t_idx
                for i in range(2):
                    U[i * 3 + 0, col] = u[i * 2 + 0]
                    U[i * 3 + 1, col] = u[i * 2 + 1]
                    U[i * 3 + 2, col] = 0.0
                for j in range(2):
                    for k_local in range(2):
                        k_global = k_offset + k_local
                        V[j * 10 + k_global, col] = v[j * 2 + k_local]
                for k_global in range(k_offset, k_offset + 2):
                    V[2 * 10 + k_global, col] = 0.0
                for k_local in range(2):
                    k_global = k_offset + k_local
                    for i in range(2):
                        W[k_global * 2 + i, col] = w[k_local * 2 + i]
        
        col = 35
        U[0 * 3 + 2, col] = 1.0
        U[1 * 3 + 2, col] = 1.0
        for k in range(10):
            V[2 * 10 + k, col] = 1.0
        for k in range(10):
            W[k * 2 + 0, col] = 1.0
            W[k * 2 + 1, col] = 1.0
        
        return U, V, W, rank
    
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    # Start with the known rank-36 solution
    U_best, V_best, W_best, rank_best = get_strassen_decomposition()
    
    # Try to find lower rank via numerical optimization
    np.random.seed(42)
    
    for target_rank in range(30, rank_best):
        if time.time() > deadline:
            break
        
        found = False
        for restart in range(5):
            if time.time() > deadline:
                break
            
            U = np.random.randn(dim_u, target_rank) * 0.5
            V = np.random.randn(dim_v, target_rank) * 0.5
            W = np.random.randn(dim_w, target_rank) * 0.5
            
            lr = 0.05
            for step in range(300):
                if time.time() > deadline:
                    break
                
                T_rec = np.einsum('ur,vr,wr->uvw', U, V, W)
                Err = T - T_rec
                
                G_U = np.einsum('uvw,vr,wr->ur', Err, V, W)
                G_V = np.einsum('uvw,ur,wr->vr', Err, U, W)
                G_W = np.einsum('uvw,ur,vr->wr', Err, U, V)
                
                U -= lr * G_U
                V -= lr * G_V
                W -= lr * G_W
                
                if step % 50 == 0:
                    err_norm = np.linalg.norm(Err)
                    if err_norm < 1e-6:
                        break
            
            valid, U_r, V_r, W_r = verify(U, V, W)
            if valid:
                U_best, V_best, W_best = U_r, V_r, W_r
                rank_best = target_rank
                found = True
                break
    
    valid, U_final, V_final, W_final = verify(U_best, V_best, W_best)
    if not valid:
        U_best, V_best, W_best, rank_best = get_schoolbook()
    
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
