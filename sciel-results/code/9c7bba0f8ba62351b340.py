"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    nm = n * m   # 6
    mp = m * p   # 27
    pn = p * n   # 18
    
    # Build target tensor T[a,b,c]
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p
    
    def check_decomp(U_r, V_r, W_r, tol=1e-8):
        T_rec = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        return np.max(np.abs(T_rec - T)) < tol
    
    def try_rank(r, max_iters=50, n_restarts=10):
        for restart in range(n_restarts):
            if time.time() > deadline:
                return None
            np.random.seed(restart * 7919 + r * 104729 + 42)
            scale = 1.0 / np.sqrt(r)
            U_r = np.random.randn(nm, r) * scale
            V_r = np.random.randn(mp, r) * scale
            W_r = np.random.randn(pn, r) * scale
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Update U
                B = np.einsum('br,cr->rbc', V_r, W_r).reshape(r, mp*pn)
                T_flat = T.reshape(nm, mp*pn)
                BBT = B @ B.T + 1e-10 * np.eye(r)
                U_r = T_flat @ B.T @ np.linalg.inv(BBT)
                
                # Update V
                C = np.einsum('ar,cr->rac', U_r, W_r).reshape(r, nm*pn)
                T_flat_v = T.transpose(1,0,2).reshape(mp, nm*pn)
                CCT = C @ C.T + 1e-10 * np.eye(r)
                V_r = T_flat_v @ C.T @ np.linalg.inv(CCT)
                
                # Update W
                D = np.einsum('ar,br->rab', U_r, V_r).reshape(r, nm*mp)
                T_flat_w = T.transpose(2,0,1).reshape(pn, nm*mp)
                DDT = D @ D.T + 1e-10 * np.eye(r)
                W_r = T_flat_w @ D.T @ np.linalg.inv(DDT)
            
            if check_decomp(U_r, V_r, W_r):
                U_clean = np.round(U_r, 6)
                V_clean = np.round(V_r, 6)
                W_clean = np.round(W_r, 6)
                if check_decomp(U_clean, V_clean, W_clean):
                    return U_clean, V_clean, W_clean
                return U_r, V_r, W_r
        return None
    
    # Try ranks from lower bound up
    lower_bound = max(nm, mp, pn)  # 27
    for r in range(lower_bound, best_rank):
        if time.time() > deadline - 15:
            break
        result = try_rank(r, max_iters=40, n_restarts=5)
        if result is not None:
            U_r, V_r, W_r = result
            best_rank = r
            best_U, best_V, best_W = U_r, V_r, W_r
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
