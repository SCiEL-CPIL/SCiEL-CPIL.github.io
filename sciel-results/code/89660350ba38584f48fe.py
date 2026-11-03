import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def _build_target_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _snap_to_half_integer(x, tol=1e-6):
    x2 = x * 2
    rounded = np.round(x2)
    if np.allclose(x2, rounded, atol=tol):
        return rounded / 2.0
    return None

def _verify_decomposition(U, V, W, T, tol=1e-9):
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=tol)

def _als_step(U, V, W, T, n, m, p, max_iter=200, tol=1e-10):
    nm, mp, pn = n*m, m*p, p*n
    rank = U.shape[1]
    
    for it in range(max_iter):
        # Update U
        A = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(mp*pn, rank)
        A_pinv = np.linalg.pinv(A)
        U_new = np.zeros_like(U)
        for a in range(nm):
            U_new[a] = A_pinv @ T[a].flatten()
        U = U_new
        
        # Update V
        A = (U[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(nm*pn, rank)
        A_pinv = np.linalg.pinv(A)
        V_new = np.zeros_like(V)
        for b in range(mp):
            V_new[b] = A_pinv @ T[:, b, :].flatten()
        V = V_new
        
        # Update W
        A = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(nm*mp, rank)
        A_pinv = np.linalg.pinv(A)
        W_new = np.zeros_like(W)
        for c in range(pn):
            W_new[c] = A_pinv @ T[:, :, c].flatten()
        W = W_new
        
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(T_recon - T)
        if err < tol:
            break
    
    return U, V, W

def _try_rank(target_rank, T, n, m, p, num_restarts=30, time_limit=20):
    nm, mp, pn = n*m, m*p, p*n
    start_time = time.time()
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        np.random.seed(restart * 7919 + 42)
        U = np.random.randn(nm, target_rank) * 0.5
        V = np.random.randn(mp, target_rank) * 0.5
        W = np.random.randn(pn, target_rank) * 0.5
        
        U, V, W = _als_step(U, V, W, T, n, m, p, max_iter=200, tol=1e-12)
        
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(T_recon - T)
        
        if err < 1e-8:
            U_s = _snap_to_half_integer(U)
            V_s = _snap_to_half_integer(V)
            W_s = _snap_to_half_integer(W)
            if U_s is not None and V_s is not None and W_s is not None:
                if _verify_decomposition(U_s, V_s, W_s, T, tol=1e-9):
                    return U_s, V_s, W_s
            # Try looser snap
            U_s = _snap_to_half_integer(U, tol=1e-4)
            V_s = _snap_to_half_integer(V, tol=1e-4)
            W_s = _snap_to_half_integer(W, tol=1e-4)
            if U_s is not None and V_s is not None and W_s is not None:
                if _verify_decomposition(U_s, V_s, W_s, T, tol=1e-9):
                    return U_s, V_s, W_s
    
    return None

def _schoolbook_solution():
    n, m, p = DIMENSIONS
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

def construct():
    n, m, p = DIMENSIONS
    total_deadline = 140.0
    start_time = time.time()
    
    T = _build_target_tensor()
    
    for target_rank in range(49, 24, -1):
        if time.time() - start_time > total_deadline - 15:
            break
        
        remaining = total_deadline - (time.time() - start_time) - 10
        if remaining < 5:
            break
        
        result = _try_rank(target_rank, T, n, m, p, num_restarts=20, time_limit=min(remaining, 25))
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    U, V, W = _schoolbook_solution()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
