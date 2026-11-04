import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def _build_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = 1.0
    return T

def _try_rank(T, rank, n, m, p, max_iters=100, tol=1e-9):
    nm = n * m
    mp = m * p
    pn = p * n
    
    U = np.random.randn(nm, rank)
    V = np.random.randn(mp, rank)
    W = np.random.randn(pn, rank)
    
    T_flat = T.reshape(nm, mp * pn)
    
    for _ in range(max_iters):
        # Update U
        Z = (V[:, :, None] * W[None, :, :]).reshape(mp * pn, rank)
        try:
            U = T_flat @ np.linalg.pinv(Z.T @ Z) @ Z.T
        except:
            pass
            
        # Update V
        Z = (U[:, :, None] * W[None, :, :]).reshape(nm * pn, rank)
        T_v = T.transpose(1, 0, 2).reshape(mp, nm * pn)
        try:
            V = T_v @ np.linalg.pinv(Z.T @ Z) @ Z.T
        except:
            pass
            
        # Update W
        Z = (U[:, :, None] * V[None, :, :]).reshape(nm * mp, rank)
        T_w = T.transpose(2, 0, 1).reshape(pn, nm * mp)
        try:
            W = T_w @ np.linalg.pinv(Z.T @ Z) @ Z.T
        except:
            pass
        
        T_rec = np.einsum('al,bl,cl->abc', U, V, W)
        error = np.linalg.norm(T - T_rec) / np.linalg.norm(T)
        if error < tol:
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            T_rec_r = np.einsum('al,bl,cl->abc', U_r, V_r, W_r)
            if np.allclose(T, T_rec_r, atol=1e-9):
                return U_r, V_r, W_r
    return None

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    start_time = time.time()
    max_time = 110
    
    best_rank = n * m * p
    best_result = None
    
    # Try ranks from 60 down to 50
    # We iterate downwards to find the lowest rank quickly if it's close to 60.
    # If the minimal rank is much lower, we might not reach it, but we return the best found.
    for rank in range(n * m * p, 40, -1):
        if time.time() - start_time > max_time:
            break
        result = _try_rank(T, rank, n, m, p, max_iters=50, tol=1e-9)
        if result is not None:
            best_rank = rank
            best_result = result
            # We could break here if we assume the first found is minimal?
            # No, we should keep searching for lower ranks.
            # But if we find 59, we should try 58.
            # So we don't break.
            
    if best_result is not None:
        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
