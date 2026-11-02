import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    def build_target_tensor():
        T = np.zeros((n, m, m, p, n, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, j, k, i, k] = 1.0
        return T
    
    T = build_target_tensor()
    
    def als_step_U(U, V, W, r):
        U_new = np.zeros_like(U)
        V_flat = V.reshape(m, p, r)
        W_flat = W.reshape(p, n, r)
        for i in range(n):
            for j in range(m):
                b = T[i, j, :, :, :, :].flatten()
                A = np.einsum('jkr,kin->jkinr', V_flat, W_flat).reshape(-1, r)
                x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                U_new[i*m+j, :] = x
        return U_new
    
    def als_step_V(U, V, W, r):
        V_new = np.zeros_like(V)
        U_flat = U.reshape(n, m, r)
        W_flat = W.reshape(p, n, r)
        for j2 in range(m):
            for k in range(p):
                b = T[:, :, j2, k, :, :].flatten()
                A = np.einsum('ijr,kin->ijkir', U_flat, W_flat).reshape(-1, r)
                x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                V_new[j2*p+k, :] = x
        return V_new
    
    def als_step_W(U, V, W, r):
        W_new = np.zeros_like(W)
        U_flat = U.reshape(n, m, r)
        V_flat = V.reshape(m, p, r)
        for k2 in range(p):
            for i2 in range(n):
                b = T[:, :, :, :, i2, k2].flatten()
                A = np.einsum('ijr,jkr->ijjkr', U_flat, V_flat).reshape(-1, r)
                x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                W_new[k2*n+i2, :] = x
        return W_new
    
    def check_solution(U, V, W, r):
        residual = np.zeros((n, m, m, p, n, p))
        for r_idx in range(r):
            u = U[:, r_idx].reshape(n, m)
            v = V[:, r_idx].reshape(m, p)
            w = W[:, r_idx].reshape(p, n)
            residual += u[:, :, None, None, None, None] * v[None, None, :, :, None, None] * w[None, None, None, None, :, :]
        return np.max(np.abs(residual - T))
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_s = np.zeros((n*m, rank_school))
    V_s = np.zeros((m*p, rank_school))
    W_s = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j, col] = 1
                V_s[j*p+k, col] = 1
                W_s[k*n+i, col] = 1
                col += 1
    
    best_result = None
    
    for target_rank in range(20, 35):
        if time.time() > deadline:
            break
        
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + target_rank * 1000 + attempt)
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            W = np.random.randn(p*n, target_rank) * 0.5
            
            for iteration in range(2000):
                if time.time() > deadline:
                    break
                
                U = als_step_U(U, V, W, target_rank)
                V = als_step_V(U, V, W, target_rank)
                W = als_step_W(U, V, W, target_rank)
                
                if iteration % 200 == 0:
                    err = check_solution(U, V, W, target_rank)
                    if err < 1e-12:
                        break
            
            err = check_solution(U, V, W, target_rank)
            if err < 1e-8:
                best_result = (U, V, W)
                break
        
        if best_result is not None:
            break
    
    if best_result is None:
        U, V, W = U_s, V_s, W_s
    else:
        U, V, W = best_result
    
    rank = U.shape[1]
    U_list = [[float(U[i, j]) for j in range(rank)] for i in range(U.shape[0])]
    V_list = [[float(V[i, j]) for j in range(rank)] for i in range(V.shape[0])]
    W_list = [[float(W[i, j]) for j in range(rank)] for i in range(W.shape[0])]
    
    return {"U": U_list, "V": V_list, "W": W_list}
