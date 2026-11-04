"""Matrix multiplication 3,5,6 - rank-minimal tensor decomposition."""
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

def _verify_fast(U, V, W, n, m, p):
    """Faster verification using numpy."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    if Ua.shape != (n*m, rank) or Va.shape != (m*p, rank) or Wa.shape != (p*n, rank):
        return False
    
    # Check all entries are integers or half-integers
    for arr in [Ua, Va, Wa]:
        if not np.allclose(2*arr, np.round(2*arr), atol=1e-9):
            return False
    
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-9):
        return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
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

def _try_discrete_search(n, m, p, rank, seed=None, max_iter=50000):
    """
    Discrete local search (hill climbing) on the half-integer lattice.
    Directly searches for valid half-integer solutions.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    # Initialize with sparse random half-integers
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Sparse initialization: ~20% non-zero entries
    for arr in [U, V, W]:
        mask = rng.random(arr.shape) < 0.2
        arr[mask] = rng.choice([-1.0, -0.5, 0.5, 1.0], size=mask.sum())
    
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    def _sse():
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        return np.sum((T - 1.0)**2)
    
    def _gradients():
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        err = T - 1.0
        grad_U = np.einsum('ijk,jkr,kir->ijr', err, V3, W3)
        grad_V = np.einsum('ijk,ijr,kir->jkr', err, U3, W3)
        grad_W = np.einsum('ijk,ijr,jkr->kir', err, U3, V3)
        return grad_U, grad_V, grad_W
    
    best_sse = _sse()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if _remaining() < 2.0:
            break
        
        sse = best_sse
        if sse < 1e-12:
            break
        
        # Select a random variable to update
        var = rng.choice(['U', 'V', 'W'])
        
        if var == 'U':
            i = rng.randint(0, n-1)
            j = rng.randint(0, m-1)
            r = rng.randint(0, rank-1)
            idx = (i*m+j, r)
            arr = U
            arr3 = U3
        elif var == 'V':
            j = rng.randint(0, m-1)
            k = rng.randint(0, p-1)
            r = rng.randint(0, rank-1)
            idx = (j*p+k, r)
            arr = V
            arr3 = V3
        else:
            k = rng.randint(0, p-1)
            i = rng.randint(0, n-1)
            r = rng.randint(0, rank-1)
            idx = (k*n+i, r)
            arr = W
            arr3 = W3
        
        # Get gradient for this variable
        grad_U, grad_V, grad_W = _gradients()
        
        if var == 'U':
            g = grad_U[i, j, r]
        elif var == 'V':
            g = grad_V[j, k, r]
        else:
            g = grad_W[k, i, r]
        
        # Determine direction from gradient sign
        if g > 0:
            direction = -0.5  # Decrease to reduce SSE
        elif g < 0:
            direction = 0.5   # Increase to reduce SSE
        else:
            direction = rng.choice([-0.5, 0.5])
        
        # Try the move
        old_val = arr[idx]
        new_val = old_val + direction
        arr[idx] = new_val
        
        new_sse = _sse()
        
        if new_sse < best_sse:
            best_sse = new_sse
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        else:
            # Reject: revert
            arr[idx] = old_val
    
    if best_sse < 1e-9:
        Ua_snapped = np.round(2 * best_U) / 2.0
        Va_snapped = np.round(2 * best_V) / 2.0
        Wa_snapped = np.round(2 * best_W) / 2.0
        
        U_out = Ua_snapped.tolist()
        V_out = Va_snapped.tolist()
        W_out = Wa_snapped.tolist()
        
        if _verify_fast(U_out, V_out, W_out, n, m, p):
            return U_out, V_out, W_out
    
    return None

def _try_altsolve(n, m, p, rank, max_iter=300, seed=None, tol=1e-10):
    """
    Alternating linear solve on continuous variables, then snap.
    """
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
        
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = V[j*p + k] * W[k*n + i]
                try:
                    U[i*m + j] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, rank))
                for i in range(n):
                    M[i, :] = U[i*m + j] * W[k*n + i]
                try:
                    V[j*p + k] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U[i*m + j] * V[j*p + k]
                try:
                    W[k*n + i] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        error = np.max(np.abs(T - 1.0))
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if error < tol:
            break
    
    if best_error < tol:
        Ua_snapped = np.round(2 * best_U) / 2.0
        Va_snapped = np.round(2 * best_V) / 2.0
        Wa_snapped = np.round(2 * best_W) / 2.0
        
        U_out = Ua_snapped.tolist()
        V_out = Va_snapped.tolist()
        W_out = Wa_snapped.tolist()
        
        if _verify_fast(U_out, V_out, W_out, n, m, p):
            return U_out, V_out, W_out
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook = _schoolbook(n, m, p)
    best = schoolbook
    best_rank = n * m * p
    
    min_rank = max(n*m, m*p, n*p)
    
    for rank in range(min_rank, best_rank + 1):
        if _remaining() < 10.0:
            break
        
        found = False
        
        # Try discrete search first (more likely to find exact half-integer solutions)
        for attempt in range(5):
            if _remaining() < 5.0:
                break
            
            result = _try_discrete_search(n, m, p, rank, seed=rank*100+attempt)
            if result is not None:
                U, V, W = result
                if _verify_fast(U, V, W, n, m, p):
                    best = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    found = True
                    break
        
        # Try alternating solve if discrete didn't work
        if not found:
            for attempt in range(3):
                if _remaining() < 5.0:
                    break
                
                result = _try_altsolve(n, m, p, rank, max_iter=200, seed=rank*100+100+attempt)
                if result is not None:
                    U, V, W = result
                    if _verify_fast(U, V, W, n, m, p):
                        best = {"U": U, "V": V, "W": W}
                        best_rank = rank
                        found = True
                        break
        
        if found:
            break
    
    if not _verify_fast(best["U"], best["V"], best["W"], n, m, p):
        best = schoolbook
    
    return best
