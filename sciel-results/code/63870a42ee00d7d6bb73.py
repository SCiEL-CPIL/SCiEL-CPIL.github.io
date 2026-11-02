import numpy as np
import time

DIMENSIONS = [2, 2, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    dim_a = n * m  # 4
    dim_b = m * p  # 10
    dim_c = p * n  # 10
    
    T = np.zeros((dim_a, dim_b, dim_c))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook
    rank_sb = n * m * p
    U_sb = np.zeros((dim_a, rank_sb))
    V_sb = np.zeros((dim_b, rank_sb))
    W_sb = np.zeros((dim_c, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    def err(U, V, W):
        return np.linalg.norm(T - np.einsum('ar,br,cr->abc', U, V, W))
    
    def round_half(X):
        return np.round(X * 2) / 2
    
    for target_rank in range(best_rank - 1, 5, -1):
        if time.time() > deadline:
            break
        
        found = False
        n_restarts = 30
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(restart * 7919 + target_rank * 104729)
            U = np.random.randn(dim_a, target_rank) * 0.3
            V = np.random.randn(dim_b, target_rank) * 0.3
            W = np.random.randn(dim_c, target_rank) * 0.3
            
            prev_err = np.inf
            for it in range(300):
                if time.time() > deadline:
                    break
                
                for a in range(dim_a):
                    M = np.einsum('br,cr->bcr', V, W).reshape(-1, target_rank)
                    U[a] = np.linalg.lstsq(M, T[a].flatten(), rcond=None)[0]
                
                for b in range(dim_b):
                    M = np.einsum('ar,cr->acr', U, W).reshape(-1, target_rank)
                    V[b] = np.linalg.lstsq(M, T[:, b, :].flatten(), rcond=None)[0]
                
                for c in range(dim_c):
                    M = np.einsum('ar,br->abr', U, V).reshape(-1, target_rank)
                    W[c] = np.linalg.lstsq(M, T[:, :, c].flatten(), rcond=None)[0]
                
                e = err(U, V, W)
                if e < 1e-8:
                    Ur, Vr, Wr = round_half(U), round_half(V), round_half(W)
                    if err(Ur, Vr, Wr) < 1e-6:
                        best_rank, best_U, best_V, best_W = target_rank, Ur, Vr, Wr
                        found = True
                        break
                if e > prev_err * 0.999 and it > 50:
                    break
                prev_err = e
            
            if found:
                break
        
        if found:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
