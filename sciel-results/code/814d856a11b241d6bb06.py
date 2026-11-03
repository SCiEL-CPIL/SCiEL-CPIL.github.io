"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]
import numpy as np
import time
import random
from itertools import product

# EVOLVE-BLOCK-START
def _is_half_integer(x, tol=1e-9):
    """Check if x is close to a half-integer."""
    nearest = round(x * 2) / 2
    return abs(x - nearest) < tol

def _snap_to_half_integer(arr, tol=1e-9):
    """Snap array entries to nearest half-integer if close enough."""
    snapped = np.round(arr * 2) / 2
    if np.max(np.abs(arr - snapped)) < tol:
        return snapped, True
    return arr, False

def _check_exact(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition is exact."""
    # Reconstruct tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(len(U[0])):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _als_iterate_fast(U, V, W, n, m, p, max_iter=200, tol=1e-12):
    """Faster ALS using batch matrix operations."""
    rank = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for it in range(max_iter):
        U_old = U.copy()
        V_old = V.copy()
        W_old = W.copy()
        
        # Update U: for each (i,j)
        # We want sum_c U[idx,c] V[j*p+k,c] W[k*n+i,c] = 1 for all k
        # Let M[k,c] = V[j*p+k,c] W[k*n+i,c]. M is p x rank.
        # We want M^T u = 1_p.
        # Least squares: (M M^T) u = M 1_p.
        for i in range(n):
            for j in range(m):
                idx = i * m + j
                # Extract V rows for j: V[j*p : j*p+p, :] (p x rank)
                # Extract W rows for i: W[k*n+i] for k in 0..p-1
                # W indices: k*n + i. So W[np.arange(p)*n + i, :] (p x rank)
                M = V[j * p : j * p + p, :] * W[np.arange(p) * n + i, :]
                # M is p x rank
                A = M @ M.T # p x p
                b = M @ np.ones(rank) # p x 1 -> p
                try:
                    sol = np.linalg.lstsq(A, b, rcond=None)[0]
                    U[idx, :] = sol
                except:
                    pass
        
        # Update V: for each (j,k)
        # We want sum_c V[idx,c] U[i*m+j,c] W[k*n+i,c] = 1 for all i
        # Let M[i,c] = U[i*m+j,c] W[k*n+i,c]. M is n x rank.
        # We want M^T v = 1_n.
        # Least squares: (M M^T) v = M 1_n.
        for j in range(m):
            for k in range(p):
                idx = j * p + k
                # U indices: i*m + j for i in 0..n-1
                # W indices: k*n + i for i in 0..n-1
                M = U[np.arange(n) * m + j, :] * W[np.arange(n) * n + k, :]
                # M is n x rank
                A = M @ M.T # n x n
                b = M @ np.ones(rank) # n
                try:
                    sol = np.linalg.lstsq(A, b, rcond=None)[0]
                    V[idx, :] = sol
                except:
                    pass
        
        # Update W: for each (k,i)
        # We want sum_c W[idx,c] U[i*m+j,c] V[j*p+k,c] = 1 for all j
        # Let M[j,c] = U[i*m+j,c] V[j*p+k,c]. M is m x rank.
        # We want M^T w = 1_m.
        # Least squares: (M M^T) w = M 1_m.
        for k in range(p):
            for i in range(n):
                idx = k * n + i
                # U indices: i*m + j for j in 0..m-1
                # V indices: j*p + k for j in 0..m-1
                M = U[i * m : i * m + m, :] * V[np.arange(m) * p + k, :]
                # M is m x rank
                A = M @ M.T # m x m
                b = M @ np.ones(rank) # m
                try:
                    sol = np.linalg.lstsq(A, b, rcond=None)[0]
                    W[idx, :] = sol
                except:
                    pass
        
        # Check convergence
        if (np.max(np.abs(U - U_old)) < tol and 
            np.max(np.abs(V - V_old)) < tol and 
            np.max(np.abs(W - W_old)) < tol):
            break
    
    return U, V, W

def _compute_error(U, V, W, n, m, p):
    """Compute maximum error of the decomposition."""
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _structured_init(n, m, p, rank, seed=None):
    """Create structured initialization based on known patterns."""
    if seed is not None:
        np.random.seed(seed)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strategy 1: Try to use rank = n*m*p with sparse structure
    if rank >= n * m * p:
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        # Add random perturbation for remaining columns
        for c in range(col, rank):
            U[:, c] = np.random.randn(n * m) * 0.1
            V[:, c] = np.random.randn(m * p) * 0.1
            W[:, c] = np.random.randn(p * n) * 0.1
        return U, V, W
    
    # Strategy 2: For lower ranks, use a mixture of structured and random
    # Initialize with small random values
    U = np.random.randn(n * m, rank) * 0.5
    V = np.random.randn(m * p, rank) * 0.5
    W = np.random.randn(p * n, rank) * 0.5
    
    # Add some structure: make certain entries larger
    for c in range(min(rank, n * m)):
        idx = c % (n * m)
        U[idx, c] += 1.0
    
    return U, V, W

def _polish(U, V, W, n, m, p, max_iter=50, tol=1e-14):
    """Final high-precision polish with tighter tolerance."""
    rank = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for it in range(max_iter):
        U_old = U.copy()
        V_old = V.copy()
        W_old = W.copy()
        
        # Update U
        for i in range(n):
            for j in range(m):
                idx = i * m + j
                M = V[j * p : j * p + p, :] * W[np.arange(p) * n + i, :]
                A = M @ M.T
                b = M @ np.ones(rank)
                try:
                    sol = np.linalg.lstsq(A, b, rcond=None)[0]
                    U[idx, :] = sol
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                idx = j * p + k
                M = U[np.arange(n) * m + j, :] * W[np.arange(n) * n + k, :]
                A = M @ M.T
                b = M @ np.ones(rank)
                try:
                    sol = np.linalg.lstsq(A, b, rcond=None)[0]
                    V[idx, :] = sol
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                idx = k * n + i
                M = U[i * m : i * m + m, :] * V[np.arange(m) * p + k, :]
                A = M @ M.T
                b = M @ np.ones(rank)
                try:
                    sol = np.linalg.lstsq(A, b, rcond=None)[0]
                    W[idx, :] = sol
                except:
                    pass
        
        if (np.max(np.abs(U - U_old)) < tol and 
            np.max(np.abs(V - V_old)) < tol and 
            np.max(np.abs(W - W_old)) < tol):
            break
    
    return U, V, W

def _try_rank(n, m, p, rank, max_restarts=20, seed=None):
    """Try to find a rank-r exact decomposition with multiple strategies."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    best_err = float('inf')
    best_result = None
    
    for restart in range(max_restarts):
        # Use structured initialization for some restarts, random for others
        if restart % 3 == 0:
            U, V, W = _structured_init(n, m, p, rank, seed=seed + restart)
        else:
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
        
        # ALS iterations
        U, V, W = _als_iterate_fast(U, V, W, n, m, p, max_iter=100)
        
        # Polish if close
        err = _compute_error(U, V, W, n, m, p)
        if err < best_err:
            best_err = err
            best_result = (U.copy(), V.copy(), W.copy())
        
        if err < 1e-8:
            # Polish
            U, V, W = _polish(U, V, W, n, m, p, max_iter=50)
            err = _compute_error(U, V, W, n, m, p)
            
            if err < 1e-10:
                # Try to snap to half-integers
                for snap_tol in [1e-6, 1e-4, 1e-3]:
                    U_snapped, u_ok = _snap_to_half_integer(U, tol=snap_tol)
                    V_snapped, v_ok = _snap_to_half_integer(V, tol=snap_tol)
                    W_snapped, w_ok = _snap_to_half_integer(W, tol=snap_tol)
                    if u_ok and v_ok and w_ok:
                        if _check_exact(U_snapped, V_snapped, W_snapped, n, m, p, tol=1e-9):
                            return U_snapped, V_snapped, W_snapped
                # If no snap works, return the polished version
                return U, V, W
    
    if best_result is not None:
        U, V, W = best_result
        err = _compute_error(U, V, W, n, m, p)
        if err < 1e-6:
            # Try to snap
            for snap_tol in [1e-5, 1e-4, 1e-3]:
                U_snapped, u_ok = _snap_to_half_integer(U, tol=snap_tol)
                V_snapped, v_ok = _snap_to_half_integer(V, tol=snap_tol)
                W_snapped, w_ok = _snap_to_half_integer(W, tol=snap_tol)
                if u_ok and v_ok and w_ok:
                    if _check_exact(U_snapped, V_snapped, W_snapped, n, m, p, tol=1e-9):
                        return U_snapped, V_snapped, W_snapped
            return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Schoolbook as fallback
    schoolbook_rank = n * m * p
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    # Try to find lower rank
    # Known lower bound for all-ones tensor (n,m,p) is at least max(n,m,p)
    # Try ranks from a reasonable lower bound upward
    lower_bound = max(n, m, p)
    
    best_result = None
    for rank in range(lower_bound, schoolbook_rank + 1):
        if time.time() > deadline:
            break
        result = _try_rank(n, m, p, rank, max_restarts=15, seed=rank * 1000)
        if result is not None:
            best_result = result
            break
    
    if best_result is None:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    U, V, W = best_result
    
    # Ensure the result is valid (half-integers and exact)
    # If it's not, we should probably fall back to schoolbook to be safe, 
    # or try to snap more aggressively. 
    # Given the strict constraint, if snap fails, it's better to return schoolbook 
    # (which is rank n*m*p) than an invalid lower rank.
    
    # Try to snap to half-integers
    U_snapped, u_ok = _snap_to_half_integer(U, tol=1e-6)
    V_snapped, v_ok = _snap_to_half_integer(V, tol=1e-6)
    W_snapped, w_ok = _snap_to_half_integer(W, tol=1e-6)
    
    if u_ok and v_ok and w_ok:
        if _check_exact(U_snapped, V_snapped, W_snapped, n, m, p, tol=1e-9):
            return {"U": U_snapped.tolist(), "V": V_snapped.tolist(), "W": W_snapped.tolist()}
        
    # If snapping failed or check failed, try a slightly looser snap?
    # Or just return schoolbook to guarantee validity.
    # The task says "invalid result ... scores 0".
    # Schoolbook is always valid.
    
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
