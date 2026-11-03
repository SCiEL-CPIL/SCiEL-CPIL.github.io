import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n  # 6, 18, 12
    
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    rank_sb = 36
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    
    def verify(U, V, W):
        for arr in [U, V, W]:
            doubled = arr * 2
            if not np.allclose(doubled, np.round(doubled), atol=1e-8):
                return False
        T_recon = np.einsum('at,bt,ct->abc', U, V, W)
        return np.allclose(T_recon, T, atol=1e-8)
    
    def refine(U, V, W, max_iters=25):
        r = U.shape[1]
        T1 = T.reshape(nm, mp * pn)
        T2 = T.transpose(1,0,2).reshape(mp, nm * pn)
        T3 = T.transpose(2,0,1).reshape(pn, nm * mp)
        
        for it in range(max_iters):
            if time.time() - start_time > 100:
                break
            
            # Solve for U: M_U.T @ U.T = T1.T
            M_U = np.zeros((r, mp * pn))
            for b in range(mp):
                for c in range(pn):
                    M_U[:, b*pn+c] = V[b, :] * W[c, :]
            U.T_sol = np.linalg.lstsq(M_U.T, T1.T, rcond=None)[0]
            U_new = U.T_sol.T
            
            # Solve for V: M_V.T @ V.T = T2.T
            M_V = np.zeros((r, nm * pn))
            for a in range(nm):
                for c in range(pn):
                    M_V[:, a*pn+c] = U_new[a, :] * W[c, :]
            V.T_sol = np.linalg.lstsq(M_V.T, T2.T, rcond=None)[0]
            V_new = V.T_sol.T
            
            # Solve for W: M_W.T @ W.T = T3.T
            M_W = np.zeros((r, nm * mp))
            for a in range(nm):
                for b in range(mp):
                    M_W[:, a*mp+b] = U_new[a, :] * V_new[b, :]
            W.T_sol = np.linalg.lstsq(M_W.T, T3.T, rcond=None)[0]
            W_new = W.T_sol.T
            
            U, V, W = U_new, V_new, W_new
            
            err = np.max(np.abs(T - np.einsum('at,bt,ct->abc', U, V, W)))
            if err < 1e-10:
                break
        
        return U, V, W
    
    np.random.seed(42)
    
    best = (U_sb, V_sb, W_sb)
    best_rank = rank_sb
    
    for target_rank in range(rank_sb - 1, 24, -1):
        if time.time() - start_time > 90:
            break
        found = False
        for attempt in range(12):
            if time.time() - start_time > 90:
                break
            
            if attempt == 0:
                U = U_sb[:, :target_rank].copy() + np.random.randn(nm, target_rank) * 1e-3
                V = V_sb[:, :target_rank].copy() + np.random.randn(mp, target_rank) * 1e-3
                W = W_sb[:, :target_rank].copy() + np.random.randn(pn, target_rank) * 1e-3
            elif attempt == 1:
                U = U_sb[:, :target_rank].copy()
                V = V_sb[:, :target_rank].copy()
                W = W_sb[:, :target_rank].copy()
            else:
                U = np.random.randn(nm, target_rank)
                V = np.random.randn(mp, target_rank)
                W = np.random.randn(pn, target_rank)
            
            U, V, W = refine(U, V, W)
            
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            if verify(U_r, V_r, W_r):
                best = (U_r, V_r, W_r)
                best_rank = target_rank
                found = True
                break
            if verify(U, V, W):
                best = (U, V, W)
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    return {"U": best[0].tolist(), "V": best[1].tolist(), "W": best[2].tolist()}
