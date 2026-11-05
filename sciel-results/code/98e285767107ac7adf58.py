import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 110
    
    n, m, p = 3, 3, 7
    
    def find_low_rank_3x3x3(max_rank=27, min_rank=19):
        """Find low-rank decomposition for 3x3x3 matrix multiplication."""
        n3, m3, p3 = 3, 3, 3
        dim_u = 9
        dim_v = 9
        dim_w = 9
        
        T = np.zeros((dim_u, dim_v, dim_w))
        for i in range(n3):
            for j in range(m3):
                for k in range(p3):
                    T[i*m3+j, j*p3+k, k*n3+i] = 1.0
        
        for rank in range(min_rank, max_rank + 1):
            if time.time() - start_time > DEADLINE - 30:
                break
            
            for trial in range(20):
                if time.time() - start_time > DEADLINE - 30:
                    break
                
                np.random.seed(42 + trial)
                U = np.random.randn(dim_u, rank)
                V = np.random.randn(dim_v, rank)
                W = np.random.randn(dim_w, rank)
                
                for it in range(300):
                    if time.time() - start_time > DEADLINE - 30:
                        break
                    
                    for a in range(dim_u):
                        M_a = (V[:, :, None] * W[:, None, :]).reshape(dim_v*dim_w, rank)
                        u, _, _, _ = np.linalg.lstsq(M_a, T[a].flatten(), rcond=None)
                        U[a] = u
                    
                    for b in range(dim_v):
                        M_b = (W[:, :, None] * U[:, None, :]).reshape(dim_w*dim_u, rank)
                        v, _, _, _ = np.linalg.lstsq(M_b, T[:, b].transpose().flatten(), rcond=None)
                        V[b] = v
                    
                    for c in range(dim_w):
                        M_c = (U[:, :, None] * V[:, None, :]).reshape(dim_u*dim_v, rank)
                        w, _, _, _ = np.linalg.lstsq(M_c, T[:, :, c].transpose().flatten(), rcond=None)
                        W[c] = w
                    
                    if it % 100 == 0:
                        T_r = np.einsum('ar,br,cr->abc', U, V, W)
                        if np.max(np.abs(T - T_r)) < 1e-10:
                            break
                
                T_r = np.einsum('ar,br,cr->abc', U, V, W)
                res = np.max(np.abs(T - T_r))
                if res < 1e-8:
                    return U, V, W, rank
        
        return None, None, None, None
    
    U19, V19, W19, r19 = find_low_rank_3x3x3()
    
    if r19 is not None:
        total_rank = r19 + r19 + 9
        U_full = np.zeros((9, total_rank))
        V_full = np.zeros((21, total_rank))
        W_full = np.zeros((21, total_rank))
        
        U_full[:, 0:r19] = U19
        V_full[0:9, 0:r19] = V19
        W_full[0:9, 0:r19] = W19
        
        U_full[:, r19:2*r19] = U19
        V_full[9:18, r19:2*r19] = V19
        W_full[9:18, r19:2*r19] = W19
        
        for i in range(3):
            for j in range(3):
                col = 2*r19 + i*3 + j
                U_full[i*3+j, col] = 1
                V_full[j*3+6, col] = 1
                W_full[6*3+i, col] = 1
        
        T_recon = np.einsum('ar,br,cr->abc', U_full, V_full, W_full)
        T_target = np.zeros((9, 21, 21))
        for i in range(3):
            for j in range(3):
                for k in range(7):
                    T_target[i*3+j, j*3+k, k*3+i] = 1.0
        
        if np.max(np.abs(T_target - T_recon)) < 1e-8:
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Schoolbook fallback
    rank = 63
    U = np.zeros((9, rank))
    V = np.zeros((21, rank))
    W = np.zeros((21, rank))
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(7):
                U[i*3+j, col] = 1
                V[j*3+k, col] = 1
                W[k*3+i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
