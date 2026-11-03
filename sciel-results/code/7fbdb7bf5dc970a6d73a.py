"""Matrix multiplication 2x4 by 4x5 using block-wise Strassen decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 5]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x5 matrix.
    
    Strategy:
    - Start with the rank-36 Strassen decomposition.
    - Attempt to reduce rank by finding linear dependencies among the rank-1 tensors.
    - If a dependency is found, return a lower-rank decomposition.
    
    Total rank: 36 (or lower if a dependency is found).
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    # Deadline guard
    _start = time.time()
    _deadline = _start + 140.0
    
    # Build the rank-36 decomposition
    U, V, W = _build_strassen_decomposition(n, m, p)
    rank = U[0].__len__()
    
    # Try to find linear dependencies to reduce rank
    try:
        U_reduced, V_reduced, W_reduced, new_rank = _try_reduce_rank(U, V, W, n, m, p, _start, _deadline)
        if new_rank < rank:
            return {"U": U_reduced, "V": V_reduced, "W": W_reduced}
    except Exception:
        pass
    
    return {"U": U, "V": V, "W": W}

def _build_strassen_decomposition(n, m, p):
    """Build the rank-36 Strassen decomposition."""
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    
    col = 0
    
    def add_strassen_222(i0, j0, k0, col_start):
        """
        Add a Strassen 2x2x2 decomposition for:
        C[i0:i0+2, k0:k0+2] = A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        """
        nonlocal col
        c = col_start
        
        # M1 = (A00 + A11) * (B00 + B11)
        U[i0 * m + j0][c] += 1
        U[(i0+1) * m + (j0+1)][c] += 1
        V[j0 * p + k0][c] += 1
        V[(j0+1) * p + (k0+1)][c] += 1
        W[k0 * n + i0][c] += 1
        W[(k0+1) * n + (i0+1)][c] += 1
        c += 1
        
        # M2 = (A10 + A11) * B00
        U[(i0+1) * m + j0][c] += 1
        U[(i0+1) * m + (j0+1)][c] += 1
        V[j0 * p + k0][c] += 1
        W[k0 * n + (i0+1)][c] += 1
        W[(k0+1) * n + (i0+1)][c] -= 1
        c += 1
        
        # M3 = A00 * (B01 - B11)
        U[i0 * m + j0][c] += 1
        V[j0 * p + (k0+1)][c] += 1
        V[(j0+1) * p + (k0+1)][c] -= 1
        W[(k0+1) * n + i0][c] += 1
        W[(k0+1) * n + (i0+1)][c] += 1
        c += 1
        
        # M4 = A11 * (B10 - B00)
        U[(i0+1) * m + (j0+1)][c] += 1
        V[(j0+1) * p + k0][c] += 1
        V[j0 * p + k0][c] -= 1
        W[k0 * n + i0][c] += 1
        W[k0 * n + (i0+1)][c] += 1
        c += 1
        
        # M5 = (A00 + A01) * B11
        U[i0 * m + j0][c] += 1
        U[i0 * m + (j0+1)][c] += 1
        V[(j0+1) * p + (k0+1)][c] += 1
        W[k0 * n + i0][c] -= 1
        W[(k0+1) * n + i0][c] += 1
        c += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        U[(i0+1) * m + j0][c] += 1
        U[i0 * m + j0][c] -= 1
        V[j0 * p + k0][c] += 1
        V[j0 * p + (k0+1)][c] += 1
        W[(k0+1) * n + (i0+1)][c] += 1
        c += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        U[i0 * m + (j0+1)][c] += 1
        U[(i0+1) * m + (j0+1)][c] -= 1
        V[(j0+1) * p + k0][c] += 1
        V[(j0+1) * p + (k0+1)][c] += 1
        W[k0 * n + i0][c] += 1
        c += 1
        
        col = c
        return c
    
    def add_schoolbook_2x4x1(i0, j0, k0, col_start):
        """
        Add schoolbook decomposition for:
        C[i0:i0+2, k0:k0+1] = A[i0:i0+2, j0:j0+4] * B[j0:j0+4, k0:k0+1]
        """
        nonlocal col
        c = col_start
        for di in range(2):
            for dj in range(4):
                U[(i0 + di) * m + (j0 + dj)][c] = 1
                V[(j0 + dj) * p + k0][c] = 1
                W[k0 * n + (i0 + di)][c] = 1
                c += 1
        col = c
        return c
    
    # Block 1: C[:, 0:2] from A[:, 0:2] * B[0:2, 0:2]
    col = add_strassen_222(0, 0, 0, 0)
    
    # Block 2: C[:, 0:2] from A[:, 2:4] * B[2:4, 0:2]
    col = add_strassen_222(0, 2, 0, col)
    
    # Block 3: C[:, 2:4] from A[:, 0:2] * B[0:2, 2:4]
    col = add_strassen_222(0, 0, 2, col)
    
    # Block 4: C[:, 2:4] from A[:, 2:4] * B[2:4, 2:4]
    col = add_strassen_222(0, 2, 2, col)
    
    # Block 5: C[:, 4] from A * B[:, 4] (schoolbook 2x4 * 4x1)
    col = add_schoolbook_2x4x1(0, 0, 4, col)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p, start_time, deadline):
    """
    Try to find a linear dependency among the rank-1 tensors to reduce rank.
    Uses SVD to detect null space, then attempts to find a valid lower-rank decomposition.
    
    Returns (U_reduced, V_reduced, W_reduced, new_rank) if successful,
    otherwise raises an exception.
    """
    rank = U[0].__len__()
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the tensor matrix: each column r is the vectorized rank-1 tensor T_r
    # T_r[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    tensor_matrix = np.zeros((n * m * p, rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(rank):
                    tensor_matrix[idx, r] = U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                idx += 1
    
    # Check if the columns are linearly dependent
    try:
        U_svd, s, Vt = np.linalg.svd(tensor_matrix, full_matrices=False)
        tol = 1e-10
        null_dim = sum(1 for sv in s if sv < tol)
        
        if null_dim > 0:
            # There is a linear dependency. The null space is spanned by the last null_dim rows of Vt.
            # We need to find a subset of columns that can be eliminated while preserving the decomposition.
            # Strategy: for each null space vector, identify which rank-1 terms can be removed.
            
            # Try to find a valid rank reduction by checking all single-removal candidates
            for r_drop in range(rank):
                if time.time() > deadline - 10:
                    break
                
                mask = [r for r in range(rank) if r != r_drop]
                U_red = U_arr[:, mask]
                V_red = V_arr[:, mask]
                W_red = W_arr[:, mask]
                
                if _verify_decomposition(U_red, V_red, W_red, n, m, p):
                    return U_red.tolist(), V_red.tolist(), W_red.tolist(), rank - 1
            
            # Try double removals
            for r1 in range(rank):
                if time.time() > deadline - 10:
                    break
                for r2 in range(r1 + 1, rank):
                    mask = [r for r in range(rank) if r != r1 and r != r2]
                    U_red = U_arr[:, mask]
                    V_red = V_arr[:, mask]
                    W_red = W_arr[:, mask]
                    
                    if _verify_decomposition(U_red, V_red, W_red, n, m, p):
                        return U_red.tolist(), V_red.tolist(), W_red.tolist(), rank - 2
        
    except Exception:
        pass
    
    raise ValueError("No rank reduction found")

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the factor matrices correctly decompose the matrix multiplication tensor."""
    if not U or not U[0]:
        return False
    rank = len(U[0])
    
    # Verify with multiple random test matrices
    for seed in [42, 123, 777]:
        np.random.seed(seed)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute C using the decomposition
        C_computed = np.zeros((n, p))
        for r in range(rank):
            # For each rank-1 term, compute the outer product contribution
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        C_computed[i, k] += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    
    return True
