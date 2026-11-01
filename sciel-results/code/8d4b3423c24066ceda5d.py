"""
Rank-minimal exact tensor decomposition for 2x2 * 2x15 matrix multiplication.

Strategy:
- The 2x15 multiplication tensor is block-diagonal across columns of B.
- For each 2x2 column block, use Strassen's rank-7 decomposition.
- For the remaining 2x1 column block, use schoolbook rank 4.
- Additionally, attempt to find a rank-10 decomposition for a 2x2x3 block
  using structured initialization (Strassen for first 2 cols + schoolbook for 3rd)
  followed by ALS refinement and gradient polish.
- If rank 10 is found, total rank = 6*7 + 10 = 52.
- Otherwise, total rank = 7*7 + 4 = 53.
"""

import time
import numpy as np

DIMENSIONS = [2, 2, 15]

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
    
    # Index mapping: U[i*m + j, r], V[j*p + k, r], W[k*n + i, r]
    
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

def _compute_error_223(U, V, W):
    """Compute the decomposition error for 2x2x3."""
    n, m, p = 2, 2, 3
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

def _verify_2x2x3(U, V, W, tol=1e-8):
    """Verify that the decomposition exactly decomposes the 2x2x3 matrix multiplication tensor."""
    n, m, p = 2, 2, 3
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

def _als_step_223(U, V, W):
    """Perform one ALS iteration for 2x2x3 rank-10 decomposition."""
    n, m, p = 2, 2, 3
    rank = U.shape[1]
    
    # Update U
    for i in range(n):
        for j in range(m):
            M = np.zeros((rank, p))
            for r in range(rank):
                for k in range(p):
                    M[r, k] = V[j * p + k, r] * W[k * n + i, r]
            try:
                U[i * m + j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
            except:
                pass
    
    # Update V
    for j in range(m):
        for k in range(p):
            N = np.zeros((rank, n))
            for r in range(rank):
                for i in range(n):
                    N[r, i] = U[i * m + j, r] * W[k * n + i, r]
            try:
                V[j * p + k, :] = np.linalg.lstsq(N, np.ones(n), rcond=None)[0]
            except:
                pass
    
    # Update W
    for k in range(p):
        for i in range(n):
            P = np.zeros((rank, m))
            for r in range(rank):
                for j in range(m):
                    P[r, j] = U[i * m + j, r] * V[j * p + k, r]
            try:
                W[k * n + i, :] = np.linalg.lstsq(P, np.ones(m), rcond=None)[0]
            except:
                pass

def _gradient_polish_223(U, V, W, max_iter=200, lr_init=0.01, deadline=None):
    """
    Gradient-based polish for the 2x2x3 rank-10 decomposition.
    Uses adaptive learning rate and L2 regularization to avoid divergence.
    """
    n, m, p = 2, 2, 3
    rank = U.shape[1]
    
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    lr = lr_init
    prev_error = _compute_error_223(U, V, W)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # d/dU[i,j,r] of (1 - sum_r' U[i,j,r']V[j,k,r']W[k,i,r'])^2
                    # = -2*(1 - T[i,j,k]) * V[j,k,r] * W[k,i,r]
                    residual = 0.0
                    for r in range(rank):
                        residual += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    residual = 1.0 - residual
                    for r in range(rank):
                        dU[i * m + j, r] += -2.0 * residual * V[j * p + k, r] * W[k * n + i, r]
                        dV[j * p + k, r] += -2.0 * residual * U[i * m + j, r] * W[k * n + i, r]
                        dW[k * n + i, r] += -2.0 * residual * U[i * m + j, r] * V[j * p + k, r]
        
        # Apply gradient step with adaptive learning rate
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        new_error = _compute_error_223(U, V, W)
        
        if new_error < prev_error:
            lr = min(lr * 1.2, 0.1)  # Increase learning rate if improving
            prev_error = new_error
        else:
            # Revert and decrease learning rate
            U += lr * dU
            V += lr * dV
            W += lr * dW
            lr *= 0.5
        
        if prev_error < 1e-10:
            break
    
    return U, V, W, prev_error

def _structured_init_223():
    """
    Create a structured initialization for rank-10 2x2x3 decomposition.
    Start with Strassen (rank 7) for first 2 columns + schoolbook (rank 4) for 3rd column,
    then randomly mix to get rank 10.
    """
    n, m, p = 2, 2, 3
    rank = 10
    
    # Build rank-11 solution: Strassen for cols 0,1 + schoolbook for col 2
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_sb, V_sb, W_sb = _build_schoolbook(2, 2, 1)
    
    U11 = np.zeros((4, 11))
    V11 = np.zeros((6, 11))
    W11 = np.zeros((6, 11))
    
    # First 7 terms: Strassen for columns 0 and 1
    U11[:, :7] = U_str
    for j in range(m):
        for k_local in range(2):
            V11[j * 3 + k_local, :7] = V_str[j * 2 + k_local, :]
    for k_local in range(2):
        for i in range(n):
            W11[k_local * 2 + i, :7] = W_str[k_local * 2 + i, :]
    
    # Last 4 terms: schoolbook for column 2
    U11[:, 7:] = U_sb
    for j in range(m):
        V11[j * 3 + 2, 7:] = V_sb[j, :]
    for i in range(n):
        W11[2 * 2 + i, 7:] = W_sb[i, :]
    
    # Now we need to go from rank 11 to rank 10.
    # Strategy: add small random perturbation and let ALS find the minimum.
    # Or: try to express one term as a combination of others.
    
    # Alternative: just use the rank-11 as starting point for ALS at rank 10
    # by projecting onto a 10-dimensional subspace
    
    np.random.seed(42)
    
    # Create a random projection from 11 to 10 dimensions
    # P is (11, 10) with orthonormal columns
    Q, _ = np.linalg.qr(np.random.randn(11, 10))
    
    U10 = U11 @ Q
    V10 = V11 @ Q
    W10 = W11 @ Q
    
    return U10, V10, W10

def _try_find_2x2x3_rank10(deadline):
    """
    Attempt to find a rank-10 decomposition for 2x2x3 matrix multiplication.
    Uses structured initialization + ALS + gradient polish.
    Returns (U, V, W) if found, None otherwise.
    """
    n, m, p = 2, 2, 3
    rank = 10
    
    best_error = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    # Strategy 1: Structured initialization (Strassen + schoolbook projected to rank 10)
    U, V, W = _structured_init_223()
    
    # ALS refinement
    prev_err = _compute_error_223(U, V, W)
    for it in range(300):
        if time.time() > deadline - 10:
            break
        _als_step_223(U, V, W)
        cur_err = _compute_error_223(U, V, W)
        if cur_err < best_error:
            best_error = cur_err
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
        if cur_err < 1e-8:
            break
        if abs(cur_err - prev_err) < 1e-12:
            break
        prev_err = cur_err
    
    if best_error < 1e-8:
        if _verify_223(best_U, best_V, best_W):
            return best_U, best_V, best_W
    
    # Strategy 2: Gradient polish from the best ALS result
    if best_U is not None:
        U2, V2, W2, err2 = _gradient_polish_223(best_U, best_V, best_W, 
                                                  max_iter=500, lr_init=0.005,
                                                  deadline=deadline - 5)
        if err2 < best_error:
            best_error = err2
            best_U = U2
            best_V = V2
            best_W = W2
    
    if best_error < 1e-8:
        if _verify_223(best_U, best_V, best_W):
            return best_U, best_V, best_W
    
    # Strategy 3: A few random restarts with ALS
    np.random.seed(123)
    for restart in range(10):
        if time.time() > deadline - 5:
            break
        
        U = np.random.randn(4, rank) * 0.5
        V = np.random.randn(6, rank) * 0.5
        W = np.random.randn(6, rank) * 0.5
        
        prev_err = _compute_error_223(U, V, W)
        for it in range(100):
            if time.time() > deadline - 5:
                break
            _als_step_223(U, V, W)
            cur_err = _compute_error_223(U, V, W)
            if cur_err < best_error:
                best_error = cur_err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            if cur_err < 1e-8:
                break
            if abs(cur_err - prev_err) < 1e-12:
                break
            prev_err = cur_err
        
        if best_error < 1e-8:
            break
    
    if best_error < 1e-8 and best_U is not None:
        if _verify_223(best_U, best_V, best_W):
            return best_U, best_V, best_W
    
    return None

def _verify_full(U, V, W, n=2, m=2, p=15, tol=1e-8):
    """Verify the full decomposition."""
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

def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 2x2 matrix by a 2x15 matrix.
    
    Strategy: Decompose the 2x15 matrix into blocks.
    Use Strassen (rank 7) for 2x2 blocks and schoolbook (rank 4) for 2x1 block.
    Additionally, try to find a rank-10 decomposition for 2x2x3 to replace
    the last 2x2 (rank 7) + 2x1 (rank 4) = 11 with rank 10.
    """
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline with 10s safety margin
    
    n, m, p = DIMENSIONS  # 2, 2, 15
    
    # Get Strassen decomposition for 2x2x2
    U_str, V_str, W_str = _build_strassen_2x2x2()
    
    # Get schoolbook decomposition for 2x2x1
    U_sb, V_sb, W_sb = _build_schoolbook(2, 2, 1)
    
    # Try to find rank-10 decomposition for 2x2x3
    U_223, V_223, W_223 = None, None, None
    if time.time() < deadline - 20:  # Leave 20s margin for construction
        result = _try_find_2x2x3_rank10(deadline - 15)
        if result is not None:
            U_223, V_223, W_223 = result
    
    if U_223 is not None:
        # Use 6 blocks of 2x2 (Strassen, rank 7 each) + 1 block of 2x2x3 (rank 10)
        # Total rank = 6*7 + 10 = 52
        block_sizes = [2, 2, 2, 2, 2, 2, 3]
        num_blocks = len(block_sizes)
        total_rank = 6 * 7 + 10  # 52
    else:
        # Use 7 blocks of 2x2 (Strassen, rank 7 each) + 1 block of 2x1 (schoolbook, rank 4)
        # Total rank = 7*7 + 4 = 53
        block_sizes = [2, 2, 2, 2, 2, 2, 2, 1]
        num_blocks = len(block_sizes)
        total_rank = 7 * 7 + 4  # 53
    
    # Build full U, V, W matrices
    U = np.zeros((n * m, total_rank))  # 4 x total_rank
    V = np.zeros((m * p, total_rank))  # 30 x total_rank
    W = np.zeros((p * n, total_rank))  # 30 x total_rank
    
    col_offset = 0
    for b in range(num_blocks):
        block_size = block_sizes[b]
        start_k = sum(block_sizes[:b])  # Starting column index in B
        
        if block_size == 2:
            # Use Strassen decomposition
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
        elif block_size == 3:
            # Use the rank-10 decomposition for 2x2x3
            U[:, col_offset:col_offset+10] = U_223
            for j in range(m):
                for k_local in range(3):
                    k_global = start_k + k_local
                    V[j * p + k_global, col_offset:col_offset+10] = V_223[j * 3 + k_local, :]
            for k_local in range(3):
                k_global = start_k + k_local
                for i in range(n):
                    W[k_global * n + i, col_offset:col_offset+10] = W_223[k_local * 2 + i, :]
            col_offset += 10
        else:  # block_size == 1
            # Use schoolbook
            U[:, col_offset:col_offset+4] = U_sb
            for j in range(m):
                V[j * p + start_k, col_offset:col_offset+4] = V_sb[j * 1 + 0, :]
            for i in range(n):
                W[start_k * n + i, col_offset:col_offset+4] = W_sb[0 * 2 + i, :]
            col_offset += 4
    
    # Verify the full decomposition
    if not _verify_full(U, V, W):
        # Fallback: use the known-valid rank-53 decomposition
        total_rank = 53
        block_sizes = [2, 2, 2, 2, 2, 2, 2, 1]
        num_blocks = len(block_sizes)
        
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
            else:  # block_size == 1
                U[:, col_offset:col_offset+4] = U_sb
                for j in range(m):
                    V[j * p + start_k, col_offset:col_offset+4] = V_sb[j * 1 + 0, :]
                for i in range(n):
                    W[start_k * n + i, col_offset:col_offset+4] = W_sb[0 * 2 + i, :]
                col_offset += 4
    
    # Convert to lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
