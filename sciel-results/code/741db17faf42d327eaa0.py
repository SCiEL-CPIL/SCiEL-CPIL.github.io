import numpy as np
import time
import random

DIMENSIONS = [3, 5, 6]
DEADLINE = 120.0
START_TIME = time.time()

def _elapsed():
    return time.time() - START_TIME

def _remaining():
    return DEADLINE - _elapsed() - 5.0

def _snap_to_half(x):
    return round(2 * x) / 2.0

def _verify_fast(U, V, W, n, m, p):
    if not U or not V or not W:
        return False
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    if Ua.shape != (n*m, rank) or Va.shape != (m*p, rank) or Wa.shape != (p*n, rank):
        return False
    
    for arr in [Ua, Va, Wa]:
        if not np.allclose(2*arr, np.round(2*arr), atol=1e-9):
            return False
    
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-9):
        return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_rank_als(n, m, p, rank, max_iter=2000, seed=None):
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    U = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(n*m)], dtype=float)
    V = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(m*p)], dtype=float)
    W = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(p*n)], dtype=float)
    
    best_error = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if _remaining() < 2.0:
            break
        
        # Update U
        for i in range(n):
            for j in range(m):
                # A[k, r] = V[j, k, r] * W[k, i, r]
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j*p + k] * W[k*n + i]
                # Solve A @ u = ones(p)
                try:
                    U[i*m + j] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                # A[i, r] = U[i, j, r] * W[k, i, r]
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i, :] = U[i*m + j] * W[k*n + i]
                try:
                    V[j*p + k] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                # A[j, r] = U[i, j, r] * V[j, k, r]
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U[i*m + j] * V[j*p + k]
                try:
                    W[k*n + i] = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        # Calculate error
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        error = np.max(np.abs(T - 1.0))
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if error < 1e-10:
            break
    
    if best_error < 1e-8:
        # Snap to half-integers
        U_snapped = np.vectorize(_snap_to_half)(best_U)
        V_snapped = np.vectorize(_snap_to_half)(best_V)
        W_snapped = np.vectorize(_snap_to_half)(best_W)
        
        U_out = U_snapped.tolist()
        V_out = V_snapped.tolist()
        W_out = W_snapped.tolist()
        
        if _verify_fast(U_out, V_out, W_out, n, m, p):
            return U_out, V_out, W_out
        
        # If snapping failed, try returning raw values if they are close enough?
        # No, must be half-integers.
        # Maybe try a small local search around the snapped values?
        # For now, return None to try another seed/rank.
        
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook = _schoolbook(n, m, p)
    best = schoolbook
    best_rank = n * m * p
    
    min_rank = max(n*m, m*p, n*p)
    
    # Try ranks from min_rank upward
    for rank in range(min_rank, best_rank + 1):
        if _remaining() < 10.0:
            break
        
        found = False
        
        # Try ALS with multiple seeds
        for attempt in range(20):
            if _remaining() < 5.0:
                break
            
            result = _try_rank_als(n, m, p, rank, max_iter=500, seed=rank*100+attempt)
            if result is not None:
                U, V, W = result
                if _verify_fast(U, V, W, n, m, p):
                    best = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    found = True
                    break
        
        if found:
            break
    
    # Final check
    if not _verify_fast(best["U"], best["V"], best["W"], n, m, p):
        best = schoolbook
        
    return best
