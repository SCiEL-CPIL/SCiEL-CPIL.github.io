"""Matrix multiplication 3,4,8 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START

def _schoolbook_decomp(n, m, p):
    """Return schoolbook decomposition for n x m * m x p -> n x p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify_decomp(U, V, W, n, m, p):
    """Verify that U, V, W form a valid exact decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if U.shape != (n * m, rank) or V.shape != (m * p, rank) or W.shape != (p * n, rank):
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _als_decompose(n, m, p, rank, max_iters=200, tol=1e-10, seed=0, deadline=None):
    """
    Alternating Least Squares to find a rank-r decomposition of the
    matrix multiplication tensor T[i,j,k] = 1.
    
    Returns (U, V, W) as numpy arrays or None if it fails.
    """
    np.random.seed(seed)
    
    # Initialize with small random values
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    prev_err = float('inf')
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            return None
        
        # Update W: for each (i, k), solve sum_r A_ik(r) * W[k*n+i, r] = 1
        # where A_ik(r) = sum_j U[i*m+j, r] * V[j*p+k, r]
        for i in range(n):
            for k in range(p):
                A_ik = np.zeros(rank)
                for j in range(m):
                    A_ik += U[i * m + j, :] * V[j * p + k, :]
                norm_sq = np.dot(A_ik, A_ik)
                if norm_sq > 1e-14:
                    W[k * n + i, :] = A_ik / norm_sq
                else:
                    W[k * n + i, :] = 0.0
        
        # Update U: for each (i, j), solve sum_r B_ij(r) * U[i*m+j, r] = 1
        # where B_ij(r) = sum_k V[j*p+k, r] * W[k*n+i, r]
        for i in range(n):
            for j in range(m):
                B_ij = np.zeros(rank)
                for k in range(p):
                    B_ij += V[j * p + k, :] * W[k * n + i, :]
                norm_sq = np.dot(B_ij, B_ij)
                if norm_sq > 1e-14:
                    U[i * m + j, :] = B_ij / norm_sq
                else:
                    U[i * m + j, :] = 0.0
        
        # Update V: for each (j, k), solve sum_r C_jk(r) * V[j*p+k, r] = 1
        # where C_jk(r) = sum_i U[i*m+j, r] * W[k*n+i, r]
        for j in range(m):
            for k in range(p):
                C_jk = np.zeros(rank)
                for i in range(n):
                    C_jk += U[i * m + j, :] * W[k * n + i, :]
                norm_sq = np.dot(C_jk, C_jk)
                if norm_sq > 1e-14:
                    V[j * p + k, :] = C_jk / norm_sq
                else:
                    V[j * p + k, :] = 0.0
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    err += (val - 1.0) ** 2
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    
    # Final check
    if err < 1e-8:
        # Try rounding to half-integers
        U_r = np.round(U * 2) / 2.0
        V_r = np.round(V * 2) / 2.0
        W_r = np.round(W * 2) / 2.0
        
        err2 = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_r[i * m + j, :], V_r[j * p + k, :] * W_r[k * n + i, :])
                    err2 += (val - 1.0) ** 2
        
        if err2 < 1e-8:
            return U_r, V_r, W_r
        
        # Try integer rounding
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        
        err3 = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_i[i * m + j, :], V_i[j * p + k, :] * W_i[k * n + i, :])
                    err3 += (val - 1.0) ** 2
        
        if err3 < 1e-8:
            return U_i, V_i, W_i
        
        # Return unrounded if error is very small
        if err < 1e-6:
            return U, V, W
    
    return None


def _try_rank(n, m, p, rank, num_trials=10, deadline=None):
    """Try to find a decomposition of given rank."""
    for trial in range(num_trials):
        if deadline is not None and time.time() > deadline:
            break
        
        result = _als_decompose(n, m, p, rank, max_iters=300, tol=1e-12, seed=trial, deadline=deadline)
        if result is not None:
            U, V, W = result
            # Final verification
            if _verify_decomp(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Schoolbook rank is 96
    schoolbook_rank = n * m * p  # 96
    
    # Try ranks below schoolbook, starting from just below and going down
    # But we need to be strategic. Let's try a few promising ranks.
    # Known results: for 2x2x2, Strassen gives 7 < 8.
    # For 3x4x8, we don't have known sub-schoolbook results.
    # Let's try a range of ranks.
    
    # Try ranks from 90 down to 70, with limited trials each
    for rank in range(schoolbook_rank - 5, 69, -1):
        if time.time() > deadline:
            break
        
        result = _try_rank(n, m, p, rank, num_trials=5, deadline=deadline)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return _schoolbook_decomp(n, m, p)

# EVOLVE-BLOCK-END
