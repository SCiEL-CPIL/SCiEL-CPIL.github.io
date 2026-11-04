"""Matrix multiplication 3,4,5 - rank-minimal tensor decomposition."""
import numpy as np
import time
from itertools import combinations

DIMENSIONS = [3, 4, 5]
n, m, p = DIMENSIONS
NMP = n * m * p  # 60

def _schoolbook():
    rank = NMP
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

def _compute_tensor(U, V, W):
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    for c in range(rank):
        uc = U[:, c].reshape(n, m)
        vc = V[:, c].reshape(m, p)
        wc = W[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += uc[i, j] * vc[j, k] * wc[k, i]
    return T

def _try_rank(U, V, W, target_rank, max_iter=1000, tol=1e-9):
    """Try alternating least squares to find rank target_rank decomposition."""
    current_rank = U.shape[1]
    if target_rank >= current_rank:
        return U, V, W, 0
    
    best_err = float('inf')
    best_U, best_V, best_W = U[:, :target_rank].copy(), V[:, :target_rank].copy(), W[:, :target_rank].copy()
    
    for it in range(max_iter):
        # Update U
        U_new = best_U.copy()
        for i in range(n):
            for j in range(m):
                A = np.zeros((target_rank, p))
                for c in range(target_rank):
                    for k in range(p):
                        A[c, k] = best_V[j * p + k, c] * best_W[k * n + i, c]
                b = np.ones(p)
                try:
                    x, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                    U_new[i * m + j, :] = x
                except:
                    pass
        best_U = U_new
        
        # Update V
        V_new = best_V.copy()
        for j in range(m):
            for k in range(p):
                A = np.zeros((target_rank, n))
                for c in range(target_rank):
                    for i in range(n):
                        A[c, i] = best_U[i * m + j, c] * best_W[k * n + i, c]
                b = np.ones(n)
                try:
                    x, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                    V_new[j * p + k, :] = x
                except:
                    pass
        best_V = V_new
        
        # Update W
        W_new = best_W.copy()
        for k in range(p):
            for i in range(n):
                A = np.zeros((target_rank, m))
                for c in range(target_rank):
                    for j in range(m):
                        A[c, j] = best_U[i * m + j, c] * best_V[j * p + k, c]
                b = np.ones(m)
                try:
                    x, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                    W_new[k * n + i, :] = x
                except:
                    pass
        best_W = W_new
        
        T = _compute_tensor(best_U, best_V, best_W)
        err = np.max(np.abs(T - 1.0))
        if err < best_err:
            best_err = err
        if err < tol:
            break
    
    if best_err < tol:
        return best_U, best_V, best_W, best_err
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    U, V, W = _schoolbook()
    current_rank = NMP
    
    # Try decreasing ranks
    for target in range(current_rank - 1, 40, -1):
        if time.time() > deadline:
            break
        result = _try_rank(U, V, W, target, max_iter=500)
        if result is not None:
            U, V, W, _ = result
            current_rank = target
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
