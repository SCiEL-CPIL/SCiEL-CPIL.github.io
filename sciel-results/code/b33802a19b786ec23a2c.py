import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 5, 5]
_DEADLINE = 135.0

def _remaining():
    return _DEADLINE - time.time()

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _verify(n, m, p, U, V, W):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _verify_fast(n, m, p, U, V, W):
    """Fast vectorized verification using numpy."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_arr[i*m+j], V_arr[j*p+k]) 
                # This isn't right - need triple product
                # s = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
                pass
    
    # Proper vectorized check
    for c in range(rank):
        pass
    
    # Just do the direct check but with numpy for speed
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _hoos_search(n, m, p, target_rank, max_iters=100, tol=1e-10):
    """
    Higher-Order Orthogonal Iteration (HOOI) for tensor decomposition.
    T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    """
    np.random.seed(42)
    
    # Initialize with random factors
    U = np.random.randn(n*m, target_rank)
    V = np.random.randn(m*p, target_rank)
    W = np.random.randn(p*n, target_rank)
    
    # Normalize
    for c in range(target_rank):
        norm = np.sqrt(np.sum(U[:,c]**2) + np.sum(V[:,c]**2) + np.sum(W[:,c]**2))
        if norm > 0:
            U[:,c] /= norm
            V[:,c] /= norm
            W[:,c] /= norm
    
    best_error = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 15:
            break
        
        # Update U: for each (i,j), U[i,j,:] should be proportional to 
        # sum_k V[j,k,:] * W[k,i,:]
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                # U_new[i*m+j, c] = sum_k V[j*p+k, c] * W[k*n+i, c]
                for k in range(p):
                    U_new[i*m+j, :] += V[j*p+k, :] * W[k*n+i, :]
        
        # Update V: for each (j,k), V[j,k,:] should be proportional to
        # sum_i U[i*m+j, :] * W[k*n+i, :]
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                for i in range(n):
                    V_new[j*p+k, :] += U[i*m+j, :] * W[k*n+i, :]
        
        # Update W: for each (k,i), W[k,i,:] should be proportional to
        # sum_j U[i*m+j, :] * V[j*p+k, :]
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                for j in range(m):
                    W_new[k*n+i, :] += U[i*m+j, :] * V[j*p+k, :]
        
        # Check convergence
        error = np.sqrt(np.sum((U_new - U)**2) + np.sum((V_new - V)**2) + np.sum((W_new - W)**2))
        
        U, V, W = U_new, V_new, W_new
        
        # Compute reconstruction error
        recon_error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(target_rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    recon_error += (s - 1.0)**2
        
        if recon_error < best_error:
            best_error = recon_error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if recon_error < tol:
            break
    
    return best_U, best_V, best_W, best_error

def _try_hooi(n, m, p):
    """Try HOOI for various ranks, starting from low."""
    if time.time() > _DEADLINE - 30:
        return None
    
    # Known bounds: for 2x5x5, try ranks from 21 upward
    for target_rank in range(21, 50):
        if time.time() > _DEADLINE - 10:
            break
        
        U, V, W, err = _hoos_search(n, m, p, target_rank, max_iters=50)
        
        if err < 1e-8:
            # Try to round to rationals
            for scale in [2, 4, 8, 16]:
                U_r = np.round(U * scale) / scale
                V_r = np.round(V * scale) / scale
                W_r = np.round(W * scale) / scale
                
                if _verify(n, m, p, U_r.tolist(), V_r.tolist(), W_r.tolist()):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            
            # Try integer rounding
            U_r = np.round(U)
            V_r = np.round(V)
            W_r = np.round(W)
            if _verify(n, m, p, U_r.tolist(), V_r.tolist(), W_r.tolist()):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    return None

def _try_algebraic_search(n, m, p):
    """
    Structured algebraic search using the fact that the multiplication tensor
    has a specific algebraic structure.
    
    Key insight: For matrix multiplication C = A*B where A is n x m and B is m x p,
    we can use the standard decomposition but look for cancellations.
    """
    if time.time() > _DEADLINE - 20:
        return None
    
    # Try to find a rank-49 solution by modifying schoolbook
    # Strategy: find two terms that can be combined or one that can be eliminated
    # using linear dependence
    
    base = _schoolbook(n, m, p)
    U = np.array(base['U'], dtype=float)
    V = np.array(base['V'], dtype=float)
    W = np.array(base['W'], dtype=float)
    
    rank = n * m * p
    target_rank = rank - 1
    
    # Try removing each term and seeing if the remaining can be adjusted
    for removed in range(rank):
        if time.time() > _DEADLINE - 10:
            break
        
        # Create reduced matrices
        U_red = np.delete(U, removed, axis=1)
        V_red = np.delete(V, removed, axis=1)
        W_red = np.delete(W, removed, axis=1)
        
        # Check if the remaining terms already satisfy the equations
        # (unlikely, but worth checking)
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(target_rank):
                        s += U_red[i*m+j, c] * V_red[j*p+k, c] * W_red[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        
        if ok:
            return {"U": U_red.tolist(), "V": V_red.tolist(), "W": W_red.tolist()}
    
    return None

def _construct():
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    
    # Try algebraic search first (fast)
    result = _try_algebraic_search(n, m, p)
    if result is not None:
        if _verify(n, m, p, result['U'], result['V'], result['W']):
            return result
    
    # Try HOOI
    result = _try_hooi(n, m, p)
    if result is not None:
        if _verify(n, m, p, result['U'], result['V'], result['W']):
            return result
    
    return best

def construct():
    try:
        return _construct()
    except Exception:
        n, m, p = DIMENSIONS
        return _schoolbook(n, m, p)
