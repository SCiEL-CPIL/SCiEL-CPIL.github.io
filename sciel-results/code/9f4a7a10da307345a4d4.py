"""Matrix multiplication 3,3,5 - find rank-minimal tensor decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify(U, V, W, n, m, p, atol=1e-9):
    """Verify that U, V, W give a valid decomposition of the multiplication tensor."""
    T = _build_target_tensor(n, m, p)
    T_recon = np.einsum('ad,bd,cd->abc', U, V, W)
    if not np.allclose(T_recon, T, atol=atol):
        return False
    return True

def _snap_to_half_integers(arr, tol=0.05):
    """Snap values close to half-integers to exact half-integers."""
    result = arr.copy()
    for idx in np.ndindex(result.shape):
        val = result[idx]
        closest = round(val * 2) / 2.0
        if abs(val - closest) < tol:
            result[idx] = closest
    return result

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _als_iteration(T, U, V, W, max_iter=200, tol=1e-12, deadline=None):
    """
    Perform ALS iterations to refine the decomposition.
    T is (A, B, C), U is (A, r), V is (B, r), W is (C, r).
    """
    A, B, C = T.shape
    r = U.shape[1]
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        # Update U
        # U[:,d] = sum_{b,c} T[a,b,c] V[b,d] W[c,d] / sum_{b,c} V[b,d]^2 W[c,d]^2
        num_U = np.einsum('abc,bd,cd->ad', T, V, W)
        den_U = np.einsum('bd,cd->d', V**2, W**2)
        den_U[den_U < 1e-12] = 1e-12
        new_U = num_U / den_U
        
        # Update V
        num_V = np.einsum('abc,ad,cd->bd', T, U, W)
        den_V = np.einsum('ad,cd->d', U**2, W**2)
        den_V[den_V < 1e-12] = 1e-12
        new_V = num_V / den_V
        
        # Update W
        num_W = np.einsum('abc,ad,bd->cd', T, U, V)
        den_W = np.einsum('ad,bd->d', U**2, V**2)
        den_W[den_W < 1e-12] = 1e-12
        new_W = num_W / den_W
        
        # Check convergence
        err_U = np.linalg.norm(new_U - U)
        err_V = np.linalg.norm(new_V - V)
        err_W = np.linalg.norm(new_W - W)
        
        U = new_U
        V = new_V
        W = new_W
        
        if err_U + err_V + err_W < tol:
            break
            
    return U, V, W

def _try_rank(n, m, p, target_rank, deadline, max_restarts=50, max_iter=300):
    """
    Try to find a rank-`target_rank` decomposition using ALS with random restarts.
    Returns (U, V, W) if found, else None.
    """
    T = _build_target_tensor(n, m, p)
    A, B, C = n * m, m * p, p * n
    
    best_U = None
    best_V = None
    best_W = None
    best_err = float('inf')
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Random initialization
        U = np.random.randn(A, target_rank)
        V = np.random.randn(B, target_rank)
        W = np.random.randn(C, target_rank)
        
        # Run ALS
        U, V, W = _als_iteration(T, U, V, W, max_iter=max_iter, deadline=deadline)
        
        # Check error
        T_recon = np.einsum('ad,bd,cd->abc', U, V, W)
        err = np.linalg.norm(T_recon - T)
        
        if err < best_err:
            best_err = err
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            
        if best_err < 1e-10:
            break
            
    if best_err < 1e-10:
        # Try to snap to half-integers
        best_U = _snap_to_half_integers(best_U)
        best_V = _snap_to_half_integers(best_V)
        best_W = _snap_to_half_integers(best_W)
        
        # Verify after snapping
        if _verify(best_U, best_V, best_W, n, m, p, atol=1e-8):
            return best_U, best_V, best_W
            
    return None

def _structured_strassen_3x3x5(n, m, p):
    """
    Attempt to construct a structured decomposition based on Strassen-like ideas.
    This is a heuristic attempt to find a low-rank decomposition.
    For 3x3x3, Strassen gives rank 7.
    For 3x3x5, we don't have a known explicit construction, but we can try
    to extend Strassen's idea or use other known algorithms.
    
    For now, this function will try a few structured initializations that might
    lead to a low-rank solution when refined by ALS.
    """
    T = _build_target_tensor(n, m, p)
    A, B, C = n * m, m * p, p * n
    
    # Try a few structured initializations
    # 1. Block diagonal structure
    # 2. Random but with symmetry
    
    # This is a placeholder for more sophisticated structured search.
    # For now, we rely on random restarts in _try_rank.
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank
    # Theoretical lower bound is at least max(n*m, m*p, p*n) = max(9, 15, 15) = 15
    # Known bounds for 3x3x3 are around 23 (actually 23 is for 2x2x2? No, Strassen is 7 for 2x2x2).
    # For 3x3x3, the best known is 23? No, it's higher.
    # Let's try ranks from 35 down to 15.
    
    # We will try a range of ranks. Since ALS is local, we might need many restarts.
    # To save time, we'll try higher ranks first with fewer restarts, and lower ranks with more.
    
    ranks_to_try = list(range(35, 14, -1))
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
            
        # Allocate more time/restarts for lower ranks if we have time
        remaining_time = deadline - time.time()
        if remaining_time < 5:
            break
            
        # Heuristic: more restarts for lower ranks
        if rank <= 20:
            max_restarts = 100
            max_iter = 500
        elif rank <= 25:
            max_restarts = 50
            max_iter = 300
        else:
            max_restarts = 20
            max_iter = 200
            
        result = _try_rank(n, m, p, rank, deadline, max_restarts=max_restarts, max_iter=max_iter)
        if result is not None:
            U, V, W = result
            best_rank = rank
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            # Continue trying lower ranks
            
    return best_result
