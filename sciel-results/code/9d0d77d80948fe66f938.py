import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor as a 3D array T[i,j,k]."""
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1
    return T

def _matricize(T, mode):
    """Matricize tensor T along the specified mode (0, 1, or 2)."""
    if mode == 0:
        return T.reshape(T.shape[0], -1).T
    elif mode == 1:
        return T.reshape(-1, T.shape[1]).T
    elif mode == 2:
        return T.reshape(-1, T.shape[2]).T

def _effective_rank(mat, tol=1e-10):
    """Compute effective numerical rank via SVD."""
    try:
        s = np.linalg.svd(mat, compute_uv=False)
        rank = np.sum(s > tol)
        return int(rank)
    except:
        return mat.shape[0]

def _try_rank_reduction(T, n, m, p, target_rank, max_iter=5000, tol=1e-8):
    """
    Try to find a rank-`target_rank` decomposition using ALS with exact rational recovery.
    Returns (U, V, W) as lists of lists with integer/half-integer entries, or None.
    """
    nmp = n * m * p
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Initialize with random factors
    np.random.seed(42)
    U = np.random.randn(nm, target_rank) * 0.1
    V = np.random.randn(mp, target_rank) * 0.1
    W = np.random.randn(pn, target_rank) * 0.1
    
    best_error = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        # Update U: for each i,j, solve for U[i*m+j, :]
        for i in range(n):
            for j in range(m):
                # T[i,j,:] = U[i*m+j,:] * (V[j*p:,:].T @ W[:, :])
                # Actually: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
                # Fix i,j: for all k, sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = T[i,j,k]
                # This is: U[i*m+j,:] @ diag_r(V[j*p+k,r]*W[k*n+i,r]) summed over k... 
                # Better: reshape. For fixed i,j, the equation is:
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                # Let A_k[r] = V[j*p+k, r] * W[k*n+i, r]
                # Then sum_r U[i*m+j, r] * A_k[r] = 1 for all k
                # This is: (A @ U[i*m+j,:].T) = ones(p) where A is p x target_rank with A[k,r] = V[j*p+k,r]*W[k*n+i,r]
                A = np.zeros((p, target_rank))
                for k in range(p):
                    for r in range(target_rank):
                        A[k, r] = V[j * p + k, r] * W[k * n + i, r]
                # Solve A @ u = ones(p) for u (target_rank,)
                try:
                    u, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(p), rcond=tol)
                    U[i * m + j, :] = u
                except:
                    pass
        
        # Update V: for each j,k
        for j in range(m):
            for k in range(p):
                # For fixed j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
                # B[i, r] = U[i*m+j, r] * W[k*n+i, r]
                B = np.zeros((n, target_rank))
                for i in range(n):
                    for r in range(target_rank):
                        B[i, r] = U[i * m + j, r] * W[k * n + i, r]
                try:
                    v, residuals, rank_b, sv = np.linalg.lstsq(B, np.ones(n), rcond=tol)
                    V[j * p + k, :] = v
                except:
                    pass
        
        # Update W: for each k,i
        for k in range(p):
            for i in range(n):
                # For fixed k,i: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
                # C[j, r] = U[i*m+j, r] * V[j*p+k, r]
                C = np.zeros((m, target_rank))
                for j in range(m):
                    for r in range(target_rank):
                        C[j, r] = U[i * m + j, r] * V[j * p + k, r]
                try:
                    w, residuals, rank_c, sv = np.linalg.lstsq(C, np.ones(m), rcond=tol)
                    W[k * n + i, :] = w
                except:
                    pass
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(target_rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    err += (val - 1.0) ** 2
        err = np.sqrt(err / (n * m * p))
        
        if err < best_error:
            best_error = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < tol:
            break
    
    if best_error > 1e-6:
        return None
    
    # Try to recover exact rational/half-integer values
    U_exact = _to_half_int(best_U)
    V_exact = _to_half_int(best_V)
    W_exact = _to_half_int(best_W)
    
    if U_exact is None or V_exact is None or W_exact is None:
        return None
    
    # Verify
    if _verify(U_exact, V_exact, W_exact, n, m, p):
        return (U_exact, V_exact, W_exact)
    
    return None

def _to_half_int(matrix):
    """Try to round matrix entries to integers or half-integers."""
    result = []
    for row in matrix:
        new_row = []
        for val in row:
            # Try nearest integer
            nearest_int = round(val)
            # Try nearest half-integer
            nearest_half = round(val * 2) / 2.0
            # Check which is closer
            if abs(val - nearest_int) < abs(val - nearest_half):
                if abs(val - nearest_int) < 1e-6:
                    new_row.append(int(nearest_int))
                else:
                    return None
            else:
                if abs(val - nearest_half) < 1e-6:
                    new_row.append(nearest_half)
                else:
                    return None
        result.append(new_row)
    return result

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reconstructs the multiplication tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(len(U[0])):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    T = _build_target_tensor(n, m, p)
    
    # Compute effective ranks of matricizations
    T0 = _matricize(T, 0)  # shape (n, m*p)
    T1 = _matricize(T, 1)  # shape (m, n*p)
    T2 = _matricize(T, 2)  # shape (p, n*m)
    
    r0 = _effective_rank(T0)
    r1 = _effective_rank(T1)
    r2 = _effective_rank(T2)
    
    # Lower bound on rank is max of these
    lower_bound = max(r0, r1, r2)
    
    # Start from lower_bound and try to find a valid decomposition
    # First, get schoolbook as fallback
    schoolbook_result = _schoolbook(n, m, p)
    
    best_result = schoolbook_result
    best_rank = n * m * p
    
    # Try ranks from lower_bound upward
    for target_rank in range(lower_bound, best_rank + 1):
        if time.time() > deadline:
            break
        
        result = _try_rank_reduction(T, n, m, p, target_rank, max_iter=2000)
        
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            best_rank = target_rank
            break  # Found the minimum rank
    
    return best_result
