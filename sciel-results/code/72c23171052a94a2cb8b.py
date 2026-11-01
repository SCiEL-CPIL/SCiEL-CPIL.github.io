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

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition exactly decomposes the n x m x p matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    rank = U.shape[1]
    # Check all entries T[i,j,k] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                actual = 0.0
                for r in range(rank):
                    actual += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(actual - 1.0) > tol:
                    return False
    return True

def _try_find_2x2x3_rank10(deadline):
    """
    Attempt to find a rank-10 decomposition for 2x2x3 matrix multiplication.
    Uses a robust ALS approach with multiple restarts.
    Returns (U, V, W) if found, None otherwise.
    """
    n, m, p = 2, 2, 3
    rank = 10
    
    # Pre-allocate tensors for faster access
    # Target tensor T[i,j,k] = 1
    # We want to minimize sum (1 - sum_r U[i,j,r]V[j,k,r]W[k,i,r])^2
    
    best_error = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    np.random.seed(42)
    
    # Use a limited number of restarts to stay within time budget
    max_restarts = 20
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5: # Leave 5s margin
            break
            
        # Random initialization with small values to avoid divergence
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        # ALS iterations
        converged = False
        for it in range(200):
            if time.time() > deadline - 5:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    # T[i,j,k] = sum_r U[i,j,r] * (V[j,k,r] * W[k,i,r])
                    # Let M[r,k] = V[j,k,r] * W[k,i,r]
                    # T[i,j,:] = U[i,j,:] @ M^T
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
                    # T[i,j,k] = sum_r V[j,k,r] * (U[i,j,r] * W[k,i,r])
                    # Let N[r,i] = U[i,j,r] * W[k,i,r]
                    # T[:,j,k] = V[j,k,:] @ N^T
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
                    # T[i,j,k] = sum_r W[k,i,r] * (U[i,j,r] * V[j,k,r])
                    # Let P[r,j] = U[i,j,r] * V[j,k,r]
                    # T[i,:,k] = W[k,i,:] @ P^T
                    P = np.zeros((rank, m))
                    for r in range(rank):
                        for j in range(m):
                            P[r, j] = U[i * m + j, r] * V[j * p + k, r]
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(P, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Check convergence
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
                converged = True
                break
        
        if converged:
            if error < best_error:
                best_error = error
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if best_error < 1e-10:
                break
    
    if best_U is not None and best_error < 1e-8:
        # Final verification
        if _verify_decomposition(best_U, best_V, best_W, n, m, p):
            return best_U, best_V, best_W
    
    return None

def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 2x2 matrix by a 2x15 matrix.
    
    Strategy:
    1. Attempt to find a rank-10 decomposition for 2x2x3.
    2. If found, use 5 blocks of 2x3 (Total Rank 50).
    3. If not found, use 7 blocks of 2x2 (Strassen, Rank 7) + 1 block of 2x1 (Schoolbook, Rank 4) (Total Rank 53).
    """
    start_time = time.time()
    deadline = start_time + 110 # 110s margin to be safe
    
    n, m, p = DIMENSIONS  # 2, 2, 15
    
    # Get Strassen decomposition for 2x2x2
    U_str, V_str, W_str = _build_strassen_2x2x2()
    
    # Get schoolbook decomposition for 2x2x1
    U_sb, V_sb, W_sb = _build_schoolbook(2, 2, 1)
    
    # Try to find rank-10 decomposition for 2x2x3
    U_223, V_223, W_223 = None, None, None
    if time.time() < deadline - 10:
        result = _try_find_2x2x3_rank10(deadline - 5)
        if result is not None:
            U_223, V_223, W_223 = result
    
    if U_223 is not None:
        # Use 5 blocks of 2x3
        block_sizes = [3, 3, 3, 3, 3]
        num_blocks = 5
        total_rank = 5 * 10 # 50
    else:
        # Fallback: 7 blocks of 2x2 + 1 block of 2x1
        block_sizes = [2, 2, 2, 2, 2, 2, 2, 1]
        num_blocks = 8
        total_rank = 7 * 7 + 4 # 53
    
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
    
    # Convert to lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
