import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    def make_tensor(n, m, p):
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    T[u_idx, v_idx, w_idx] = 1.0
        return T

    def find_decomp_344(target_rank, time_limit=20):
        n, m, p = 3, 4, 4
        T = make_tensor(n, m, p)
        
        best_U, best_V, best_W = None, None, None
        best_err = float('inf')
        
        for restart in range(50):
            if time.time() > start_time + time_limit:
                break
                
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            U /= (np.linalg.norm(U, axis=0, keepdims=True) + 1e-9)
            V /= (np.linalg.norm(V, axis=0, keepdims=True) + 1e-9)
            W /= (np.linalg.norm(W, axis=0, keepdims=True) + 1e-9)
            
            err = float('inf')
            for it in range(50):
                if time.time() > start_time + time_limit:
                    break
                    
                VW = V @ W.T
                T_flat = T.reshape(n*m, -1)
                U_new, _, _, _ = np.linalg.lstsq(VW.T, T_flat, rcond=None)
                U = U_new
                
                UW = U @ W.T
                V_new, _, _, _ = np.linalg.lstsq(UW.T, T_flat, rcond=None)
                V = V_new
                
                UV = U @ V.T
                W_new, _, _, _ = np.linalg.lstsq(UV.T, T_flat, rcond=None)
                W = W_new
                
                T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
                err = np.linalg.norm(T_recon - T)
                
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if err < 1e-8:
                    break
            
            if best_err < 1e-8:
                break

        if best_err < 1e-8:
            U_r = np.round(best_U * 2) / 2
            V_r = np.round(best_V * 2) / 2
            W_r = np.round(best_W * 2) / 2
            
            T_recon = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
            if np.allclose(T_recon, T, atol=1e-9):
                return U_r, V_r, W_r
                
        return None

    U_344, V_344, W_344 = find_decomp_344(44, time_limit=30)
    
    if U_344 is not None and time.time() < deadline:
        n, m, p = 3, 4, 8
        R_sub = 44
        R_total = 2 * R_sub
        
        U_total = np.zeros((n*m, R_total))
        V_total = np.zeros((m*p, R_total))
        W_total = np.zeros((p*n, R_total))
        
        U_total[:, :R_sub] = U_344
        U_total[:, R_sub:] = U_344
        
        V_total[:16, :R_sub] = V_344
        V_total[16:, R_sub:] = V_344
        
        W_total[:12, :R_sub] = W_344
        W_total[12:, R_sub:] = W_344
        
        U_list = U_total.tolist()
        V_list = V_total.tolist()
        W_list = W_total.tolist()
        
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
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
