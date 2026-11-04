import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    fallback = {"U": U_school, "V": V_school, "W": W_school}
    
    # Build target tensor T
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    start_time = time.time()
    deadline = start_time + 115.0
    
    # Try ranks from 35 down to 20
    for R in range(35, 19, -1):
        if time.time() > deadline - 5.0:
            break
            
        result = try_find_decomposition(T, R, n, m, p, time_limit=6.0)
        if result:
            return result
            
    return fallback

def try_find_decomposition(T, R, n, m, p, time_limit):
    start_time = time.time()
    num_restarts = 3
    max_iter = 60
    
    for restart in range(num_restarts):
        if time.time() > start_time + time_limit:
            break
            
        np.random.seed(restart)
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = np.random.randn(p * n, R)
        
        for it in range(max_iter):
            if time.time() > start_time + time_limit:
                break
                
            # Update U
            V_kron_W = np.zeros((R, m * p * p * n))
            for r in range(R):
                V_kron_W[r, :] = np.kron(V[:, r], W[:, r])
            
            T_flat = T.reshape(n * m, -1)
            try:
                U = T_flat @ np.linalg.pinv(V_kron_W.T)
            except:
                break
                
            # Update V
            U_kron_W = np.zeros((R, n * m * p * n))
            for r in range(R):
                U_kron_W[r, :] = np.kron(U[:, r], W[:, r])
            
            T_flat_v = T.transpose(1, 0, 2).reshape(m * p, -1)
            try:
                V = T_flat_v @ np.linalg.pinv(U_kron_W.T)
            except:
                break
                
            # Update W
            U_kron_V = np.zeros((R, n * m * m * p))
            for r in range(R):
                U_kron_V[r, :] = np.kron(U[:, r], V[:, r])
            
            T_flat_w = T.transpose(2, 0, 1).reshape(p * n, -1)
            try:
                W = T_flat_w @ np.linalg.pinv(U_kron_V.T)
            except:
                break
                
        # Check solution
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        error = np.linalg.norm(T_recon - T)
        
        if error < 1e-6:
            # Round to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            T_recon_rounded = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
            error_rounded = np.linalg.norm(T_recon_rounded - T)
            
            if error_rounded < 1e-9:
                return {"U": U_rounded, "V": V_rounded, "W": W_rounded}
            
    return None
