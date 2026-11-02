import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook fallback
    rank_sb = n * m * p
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
    
    def verify(U, V, W, tol=1e-9):
        result = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(result - T)) < tol
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    for target_rank in range(24, rank_sb + 1):
        if time.time() > deadline:
            break
        
        for attempt in range(30):
            if time.time() > deadline:
                break
            
            np.random.seed(98765 + attempt * 7919 + target_rank * 131)
            U = np.random.randn(nm, target_rank) * 0.1
            V = np.random.randn(mp, target_rank) * 0.1
            W = np.random.randn(pn, target_rank) * 0.1
            
            for it in range(400):
                if time.time() > deadline:
                    break
                
                VW = (V[:, :, None] * W[None, :, :]).reshape(target_rank, mp * pn)
                VWtVW = VW.T @ VW + 1e-10 * np.eye(target_rank)
                T_flat = T.reshape(nm, mp * pn)
                U = np.linalg.solve(VWtVW, VW.T @ T_flat.T).T
                
                UW = (U[:, :, None] * W[None, :, :]).reshape(target_rank, nm * pn)
                UWtUW = UW.T @ UW + 1e-10 * np.eye(target_rank)
                T_by_b = T.transpose(1, 0, 2).reshape(mp, nm * pn)
                V = np.linalg.solve(UWtUW, UW.T @ T_by_b.T).T
                
                UV = (U[:, :, None] * V[None, :, :]).reshape(target_rank, nm * mp)
                UVtUV = UV.T @ UV + 1e-10 * np.eye(target_rank)
                T_by_c = T.transpose(2, 0, 1).reshape(pn, nm * mp)
                W = np.linalg.solve(UVtUV, UV.T @ T_by_c.T).T
            
            if verify(U, V, W):
                U_hi = np.round(U * 2) / 2
                V_hi = np.round(V * 2) / 2
                W_hi = np.round(W * 2) / 2
                if verify(U_hi, V_hi, W_hi):
                    if target_rank < best_rank:
                        best_rank = target_rank
                        best_U, best_V, best_W = U_hi.copy(), V_hi.copy(), W_hi.copy()
                    break
        
        if best_rank <= target_rank:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
