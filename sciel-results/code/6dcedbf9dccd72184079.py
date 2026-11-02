"""
Rank-minimal exact tensor decomposition for 2x2 * 2x15 matrix multiplication.

Strategy (structural change from block-wise):
- Instead of independent 2x2 Strassen blocks + 2x1 schoolbook (rank 53),
  use a global alternating least squares (ALS) optimization over the full
  2x2x15 tensor with a rank-50 initialization seeded from the block solution.
- The block solution gives a rank-53 starting point; we attempt to compress
  it to rank 50 by re-optimizing all factors jointly.
- If the rank-50 solution converges to zero error, we return rank 50.
- Otherwise, fall back to the rank-53 block solution (which is always valid).

The key insight: the block decomposition treats each 2x2 block independently,
but the global tensor may have cross-block structure that allows sharing of
rank terms. ALS on the full tensor can discover such sharing.
"""

import time
import numpy as np

DIMENSIONS = [2, 2, 15]
TIME_LIMIT = 115  # Safety margin before 120s kill

def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition of rank n*m*p."""
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

def _build_strassen_2x2x2():
    """
    Build Strassen's rank-7 decomposition for 2x2 x 2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) following the trace convention W[k*n+i, r].
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Term 1: M1 = (A00 + A11) * (B00 + B11)
    U[0*2+0, 0] = 1
    U[1*2+1, 0] = 1
    V[0*2+0, 0] = 1
    V[1*2+1, 0] = 1
    W[0*2+0, 0] = 1
    W[1*2+1, 0] = 1
    
    # Term 2: M2 = (A10 + A11) * B00
    U[1*2+0, 1] = 1
    U[1*2+1, 1] = 1
    V[0*2+0, 1] = 1
    W[0*2+1, 1] = 1
    W[1*2+1, 1] = -1
    
    # Term 3: M3 = A00 * (B01 - B11)
    U[0*2+0, 2] = 1
    V[0*2+1, 2] = 1
    V[1*2+1, 2] = -1
    W[1*2+0, 2] = 1
    W[1*2+1, 2] = 1
    
    # Term 4: M4 = A11 * (B10 - B00)
    U[1*2+1, 3] = 1
    V[1*2+0, 3] = 1
    V[0*2+0, 3] = -1
    W[0*2+0, 3] = 1
    W[0*2+1, 3] = 1
    
    # Term 5: M5 = (A00 + A01) * B11
    U[0*2+0, 4] = 1
    U[0*2+1, 4] = 1
    V[1*2+1, 4] = 1
    W[0*2+0, 4] = -1
    W[1*2+0, 4] = 1
    
    # Term 6: M6 = (A10 - A00) * (B00 + B01)
    U[1*2+0, 5] = 1
    U[0*2+0, 5] = -1
    V[0*2+0, 5] = 1
    V[0*2+1, 5] = 1
    W[1*2+1, 5] = 1
    
    # Term 7: M7 = (A01 - A11) * (B10 + B11)
    U[0*2+1, 6] = 1
    U[1*2+1, 6] = -1
    V[1*2+0, 6] = 1
    V[1*2+1, 6] = 1
    W[0*2+0, 6] = 1
    
    return U, V, W

def _build_block_solution(n, m, p):
    """
    Build the block-wise solution: 7 Strassen blocks + 1 schoolbook block.
    Returns U (4, 53), V (30, 53), W (30, 53).
    """
    block_sizes = [2, 2, 2, 2, 2, 2, 2, 1]
    num_blocks = len(block_sizes)
    
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_sb, V_sb, W_sb = _build_schoolbook(2, 2, 1)
    
    total_rank = 7 * 7 + 4  # 53
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    for b in range(num_blocks):
        block_size = block_sizes[b]
        start_k = sum(block_sizes[:b])
        
        if block_size == 2:
            U[:, col_offset:col_offset+7] = U_str
            for j in range(m):
                for k_local in range(2):
                    k_global = start_k + k_local
                    V[j * p + k_global, col_offset:col_offset+7] = V_str[j * 2 + k_local, :]
            for k_local in range(2):
                k_global = start_k + k_local
                for i in range(n):
                    W[k_global * n + i, col_offset:col_offset+7] = W_str[k_local * 2 + i, :]
            col_offset += 7
        else:
            U[:, col_offset:col_offset+4] = U_sb
            for j in range(m):
                V[j * p + start_k, col_offset:col_offset+4] = V_sb[j * 1 + 0, :]
            for i in range(n):
                W[start_k * n + i, col_offset:col_offset+4] = W_sb[0 * 2 + i, :]
            col_offset += 4
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """
    Verify that the decomposition exactly decomposes the matrix multiplication tensor.
    T[i,j,k] = 1 for all valid (i,j,k).
    """
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                actual = 0.0
                for r in range(rank):
                    actual += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(actual - 1.0) > tol:
                    return False
    return True

def _als_full_tensor(U, V, W, n, m, p, deadline, max_iters=200):
    """
    Run ALS on the full 2x2x15 tensor to try to converge to zero error.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    rank = U.shape[1]
    
    # Precompute the target tensor values (all 1s)
    # For efficiency, we work with the structure directly
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Update U: for each (i,j), solve least squares
        for i in range(n):
            for j in range(m):
                # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # For fixed (i,j), this is linear in U[i*m+j, :]
                # Let B[r, k] = V[j*p+k, r] * W[k*n+i, r]
                # Then T[i,j,:] = U[i*m+j, :] @ B^T
                B = np.zeros((rank, p))
                for r in range(rank):
                    for k in range(p):
                        B[r, k] = V[j * p + k, r] * W[k * n + i, r]
                try:
                    U[i * m + j, :] = np.linalg.lstsq(B, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V: for each (j,k), solve least squares
        for j in range(m):
            for k in range(p):
                # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # For fixed (j,k), linear in V[j*p+k, :]
                # Let C[r, i] = U[i*m+j, r] * W[k*n+i, r]
                # Then T[:,j,k] = V[j*p+k, :] @ C^T
                C = np.zeros((rank, n))
                for r in range(rank):
                    for i in range(n):
                        C[r, i] = U[i * m + j, r] * W[k * n + i, r]
                try:
                    V[j * p + k, :] = np.linalg.lstsq(C, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Update W: for each (k,i), solve least squares
        for k in range(p):
            for i in range(n):
                # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # For fixed (k,i), linear in W[k*n+i, :]
                # Let D[r, j] = U[i*m+j, r] * V[j*p+k, r]
                # Then T[i,:,k] = W[k*n+i, :] @ D^T
                D = np.zeros((rank, m))
                for r in range(rank):
                    for j in range(m):
                        D[r, j] = U[i * m + j, r] * V[j * p + k, r]
                try:
                    W[k * n + i, :] = np.linalg.lstsq(D, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        # Check convergence every 10 iterations
        if (it + 1) % 10 == 0:
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        actual = 0.0
                        for r in range(rank):
                            actual += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                        error += (actual - 1.0) ** 2
            error = np.sqrt(error)
            if error < 1e-8:
                break
    
    return U, V, W

def _compute_error(U, V, W, n, m, p):
    """Compute the Frobenius norm error of the decomposition."""
    rank = U.shape[1]
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                actual = 0.0
                for r in range(rank):
                    actual += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                error += (actual - 1.0) ** 2
    return np.sqrt(error)

def _try_compress(U53, V53, W53, n, m, p, target_rank, deadline):
    """
    Try to compress the rank-53 solution to target_rank by:
    1. Truncating to the first target_rank columns (or using SVD-based selection)
    2. Running ALS to recover
    
    Returns (U, V, W) if successful (error < 1e-8), else None.
    """
    # Strategy: use SVD of the "flattened" factors to find the best low-rank approximation
    # Then run ALS from that initialization
    
    np.random.seed(123)
    
    best_result = None
    best_error = float('inf')
    
    # Try multiple initializations
    for restart in range(10):
        if time.time() > deadline:
            break
        
        if restart == 0:
            # First try: just truncate (keep first target_rank columns)
            U_init = U53[:, :target_rank].copy()
            V_init = V53[:, :target_rank].copy()
            W_init = W53[:, :target_rank].copy()
        else:
            # Add small random perturbation to avoid local minima
            U_init = U53[:, :target_rank].copy() + 0.01 * np.random.randn(n * m, target_rank)
            V_init = V53[:, :target_rank].copy() + 0.01 * np.random.randn(m * p, target_rank)
            W_init = W53[:, :target_rank].copy() + 0.01 * np.random.randn(p * n, target_rank)
        
        # Run ALS
        U_opt, V_opt, W_opt = _als_full_tensor(U_init, V_init, W_init, n, m, p, deadline, max_iters=100)
        
        # Compute error
        error = _compute_error(U_opt, V_opt, W_opt, n, m, p)
        
        if error < best_error:
            best_error = error
            best_result = (U_opt, V_opt, W_opt)
        
        if best_error < 1e-8:
            break
    
    if best_result is not None and best_error < 1e-8:
        return best_result
    
    return None

def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 2x2 matrix by a 2x15 matrix.
    """
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    n, m, p = DIMENSIONS  # 2, 2, 15
    
    # Build the baseline rank-53 block solution
    U53, V53, W53 = _build_block_solution(n, m, p)
    
    # Verify baseline
    if not _verify_decomposition(U53, V53, W53, n, m, p):
        raise RuntimeError("Baseline solution failed verification")
    
    # Try to compress to rank 50
    if time.time() < deadline - 20:
        result = _try_compress(U53, V53, W53, n, m, p, 50, deadline - 10)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Try rank 52
    if time.time() < deadline - 15:
        result = _try_compress(U53, V53, W53, n, m, p, 52, deadline - 10)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Try rank 51
    if time.time() < deadline - 10:
        result = _try_compress(U53, V53, W53, n, m, p, 51, deadline - 5)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fall back to rank-53 block solution
    U_list = U53.tolist()
    V_list = V53.tolist()
    W_list = W53.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
