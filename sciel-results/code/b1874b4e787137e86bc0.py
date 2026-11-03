import time
import numpy as np
import random

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, num_tests=5):
    """
    Verify that the decomposition exactly computes matrix multiplication.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    rank = len(U[0])
    
    # Convert to float arrays for fast computation
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    np.random.seed(12345)
    for _ in range(num_tests):
        A = np.random.randint(-3, 4, (n, m)).astype(np.float64)
        B = np.random.randint(-3, 4, (m, p)).astype(np.float64)
        C_true = A @ B
        
        A_flat = A.flatten()
        B_flat = B.flatten()
        
        # C_comp[k, i] = sum_r U[i,j] * V[j,k] * W[k,i] 
        # Wait, the trace convention is W[k*n+i, r].
        # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        C_comp = np.zeros((p, n), dtype=np.float64)
        
        # Vectorized check:
        # For each r, we add U[:,r] @ A_flat * V[:,r] @ B_flat * W[:,r]
        # But we need to be careful with indices.
        
        # Let's do it efficiently.
        # A_flat is length n*m. U[:,r] is length n*m. dot gives scalar a_r.
        # B_flat is length m*p. V[:,r] is length m*p. dot gives scalar b_r.
        # W[:,r] is length p*n.
        # C_comp += a_r * b_r * W[:,r].reshape(p, n)
        
        # We can compute all a_r and b_r at once.
        # a = U.T @ A_flat  (shape rank)
        # b = V.T @ B_flat  (shape rank)
        # C_comp = (a * b) @ W  -> shape (rank) dot (rank, p*n) -> (p*n)
        
        a = U_arr.T @ A_flat
        b = V_arr.T @ B_flat
        C_flat = (a * b) @ W_arr
        
        C_comp = C_flat.reshape((p, n)).T
        
        if not np.allclose(C_comp, C_true, atol=1e-8):
            return False
            
    return True

def _quantize_matrix(M, granularity=0.5):
    """Quantize matrix entries to nearest multiple of granularity."""
    return np.round(M / granularity) * granularity

def _als_step_fix_U(V, W, A_samples, B_samples, C_samples, U, lr=0.01, momentum=0.9):
    """Update U given V, W."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_new = U.copy()
    
    # Precompute V^T W? No, V and W depend on samples.
    # For each sample (A, B, C):
    # We want to minimize || C - sum_r (V[:,r]@B) (U[:,r]@A) W[:,r] ||^2
    # This is linear in U.
    # Let y_r = (V[:,r] @ B_flat) * W[:,r]
    # Then C_flat = sum_r y_r * (U[:,r] @ A_flat)
    # C_flat = (U.T @ A_flat) @ Y  where Y is rank x (p*n) matrix with Y[r, :] = y_r
    
    # So for each sample, we solve for U columns.
    # Actually, let's accumulate gradients or normal equations.
    
    # Let's use a simple gradient step for stability and simplicity within time limits.
    # Error E = C - C_pred
    # dL/du_{i,r} = -2 * sum_samples E_{k,i} * y_{r,sample} * A_{i,j} ... wait.
    
    # Let's stick to normal equations for each U column r, aggregated over samples.
    # C_flat = sum_r alpha_r * y_r
    # where alpha_r = U[:,r] @ A_flat
    # y_r = (V[:,r] @ B_flat) * W[:,r]
    
    # This is a linear regression for each r? No, it's coupled.
    # Let's just do gradient descent on U.
    
    grad_U = np.zeros_like(U)
    
    for A, B, C in zip(A_samples, B_samples, C_samples):
        A_flat = A.flatten()
        B_flat = B.flatten()
        C_flat = C.flatten()
        
        # a = U.T @ A_flat (rank)
        a = U.T @ A_flat
        # b = V.T @ B_flat (rank)
        b = V.T @ B_flat
        
        # C_pred_flat = (a * b) @ W
        C_pred_flat = (a * b) @ W
        
        # Error
        err = C_flat - C_pred_flat
        
        # Gradient of L = 0.5 * ||err||^2 wrt U
        # err = C - (a*b) W
        # d err / d U_{i,r} = - (b_r * W[:,r]) * (A_flat[i] if index matches) ...
        
        # Let's derive properly.
        # C_flat[k*n+i] = sum_r a_r b_r W[k*n+i, r]
        # a_r = sum_{i,j} U[i*m+j, r] A[i,j]
        
        # d C_flat[idx] / d U[idx_A, r] = b_r * W[idx, r]
        
        # So grad_U[idx_A, r] = - sum_idx err[idx] * b_r * W[idx, r]
        
        # Vectorized:
        # grad_U[:, r] = - b_r * (W[:, r] @ err)
        
        for r in range(rank):
            grad_U[:, r] += -b[r] * (W[:, r] @ err)
            
    return U_new + lr * grad_U

def _als_step_fix_V(U, W, A_samples, B_samples, C_samples, V, lr=0.01):
    """Update V given U, W."""
    n, m, p = DIMENSIONS
    rank = len(V[0])
    grad_V = np.zeros_like(V)
    
    for A, B, C in zip(A_samples, B_samples, C_samples):
        A_flat = A.flatten()
        B_flat = B.flatten()
        C_flat = C.flatten()
        
        a = U.T @ A_flat
        b = V.T @ B_flat
        
        C_pred_flat = (a * b) @ W
        err = C_flat - C_pred_flat
        
        # d C_flat[idx] / d V[idx_B, r] = a_r * W[idx, r]
        for r in range(rank):
            grad_V[:, r] += -a[r] * (W[:, r] @ err)
            
    return V + lr * grad_V

def _als_step_fix_W(U, V, A_samples, B_samples, C_samples, W, lr=0.01):
    """Update W given U, V."""
    n, m, p = DIMENSIONS
    rank = len(W[0])
    grad_W = np.zeros_like(W)
    
    for A, B, C in zip(A_samples, B_samples, C_samples):
        A_flat = A.flatten()
        B_flat = B.flatten()
        C_flat = C.flatten()
        
        a = U.T @ A_flat
        b = V.T @ B_flat
        
        C_pred_flat = (a * b) @ W
        err = C_flat - C_pred_flat
        
        # d C_flat[idx] / d W[idx, r] = a_r * b_r
        for r in range(rank):
            grad_W[:, r] += -a[r] * b[r] * err
            
    return W + lr * grad_W

def _try_als_decomposition(max_rank, time_limit, num_samples=100, seed=42):
    """
    Try to find a decomposition of rank max_rank using ALS.
    """
    np.random.seed(seed)
    n, m, p = DIMENSIONS
    
    # Generate training data
    A_samples = []
    B_samples = []
    C_samples = []
    for _ in range(num_samples):
        A = np.random.randint(-2, 3, (n, m)).astype(np.float64)
        B = np.random.randint(-2, 3, (m, p)).astype(np.float64)
        C = A @ B
        A_samples.append(A)
        B_samples.append(B)
        C_samples.append(C)
        
    # Initialize
    U = np.random.randn(n*m, max_rank) * 0.1
    V = np.random.randn(m*p, max_rank) * 0.1
    W = np.random.randn(p*n, max_rank) * 0.1
    
    # Schoolbook initialization for first n*m*p ranks if max_rank is large?
    # No, let's try random init with decay.
    
    best_U, best_V, best_W = U, V, W
    best_error = np.inf
    
    start_time = time.time()
    lr = 0.005
    
    iteration = 0
    while time.time() - start_time < time_limit:
        # Update
        U = _als_step_fix_U(V, W, A_samples, B_samples, C_samples, U, lr)
        V = _als_step_fix_V(U, W, A_samples, B_samples, C_samples, V, lr)
        W = _als_step_fix_W(U, V, A_samples, B_samples, C_samples, W, lr)
        
        # Periodic verification and error check
        if iteration % 10 == 0:
            # Check error on a small subset
            A = A_samples[0]
            B = B_samples[0]
            C = C_samples[0]
            A_flat = A.flatten()
            B_flat = B.flatten()
            a = U.T @ A_flat
            b = V.T @ B_flat
            C_pred = (a * b) @ W
            error = np.linalg.norm(C.flatten() - C_pred)
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U, V, W
                
            # Decay learning rate
            lr *= 0.95
            
            # Try quantization and verification if error is low
            if error < 0.1:
                U_q = _quantize_matrix(U, 0.5).tolist()
                V_q = _quantize_matrix(V, 0.5).tolist()
                W_q = _quantize_matrix(W, 0.5).tolist()
                
                if _verify_decomposition(U_q, V_q, W_q, n, m, p, num_tests=3):
                    return {"U": U_q, "V": V_q, "W": W_q}
                
        iteration += 1
        
    # Final attempt with best found
    U_q = _quantize_matrix(best_U, 0.5).tolist()
    V_q = _quantize_matrix(best_V, 0.5).tolist()
    W_q = _quantize_matrix(best_W, 0.5).tolist()
    
    if _verify_decomposition(U_q, V_q, W_q, n, m, p, num_tests=5):
        return {"U": U_q, "V": V_q, "W": W_q}
        
    return None

def _build_strassen_5x5():
    """
    Fallback: Build the 5x5 decomposition using 4x4 Strassen + corrections.
    This is the current strategy, included for safety.
    """
    # Re-implementing the logic from the prompt's current program to ensure availability
    # but keeping it compact.
    
    def _build_2x2_strassen():
        U = np.zeros((4, 7), dtype=np.int8)
        V = np.zeros((4, 7), dtype=np.int8)
        W = np.zeros((4, 7), dtype=np.int8)
        
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[3,0] = 1
        W[0,0] = 1; W[3,0] = 1
        
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        W[2,1] = 1
        
        U[0,2] = 1
        V[1,2] = 1; V[3,2] = -1
        W[1,2] = 1; W[3,2] = 1
        
        U[3,3] = 1
        V[0,3] = -1; V[2,3] = 1
        W[0,3] = 1; W[2,3] = 1
        
        U[0,4] = 1; U[1,4] = 1
        V[3,4] = 1
        W[0,4] = -1; W[1,4] = 1
        
        U[0,5] = -1; U[2,5] = 1
        V[0,5] = 1; V[1,5] = 1
        W[3,5] = 1
        
        U[1,6] = 1; U[3,6] = -1
        V[2,6] = 1; V[3,6] = 1
        W[0,6] = 1
        
        return U, V, W

    U2, V2, W2 = _build_2x2_strassen()
    U4 = np.zeros((16, 49), dtype=np.int8)
    V4 = np.zeros((16, 49), dtype=np.int8)
    W4 = np.zeros((16, 49), dtype=np.int8)
    
    big_terms = [
        ([0, 3], [1, 1], [0, 3], [1, 1]),
        ([2, 3], [1, 1], [0], [1]),
        ([0], [1], [1, 3], [1, -1]),
        ([3], [1], [2, 0], [1, -1]),
        ([0, 1], [1, 1], [3], [1]),
        ([2, 0], [1, -1], [0, 1], [1, 1]),
        ([1, 3], [1, -1], [2, 3], [1, 1]),
    ]
    
    for b in range(7):
        A_blocks, A_coeffs, B_blocks, B_coeffs = big_terms[b]
        for s in range(7):
            r = b * 7 + s
            for i in range(4):
                for j in range(4):
                    block_idx = (i // 2) * 2 + (j // 2)
                    local_i = i % 2
                    local_j = j % 2
                    if block_idx in A_blocks:
                        coeff = A_coeffs[A_blocks.index(block_idx)]
                        U4[i * 4 + j, r] = coeff * U2[local_i * 2 + local_j, s]
            for j in range(4):
                for k in range(4):
                    block_idx = (j // 2) * 2 + (k // 2)
                    local_j = j % 2
                    local_k = k % 2
                    if block_idx in B_blocks:
                        coeff = B_coeffs[B_coeffs.index(block_idx) if False else B_blocks.index(block_idx)]
                        V4[j * 4 + k, r] = coeff * V2[local_j * 2 + local_k, s]
            
            c_contrib = {
                0: [(0, 1), (3, 1), (4, -1), (6, 1)],
                1: [(2, 1), (4, 1)],
                2: [(1, 1), (3, 1)],
                3: [(0, 1), (1, -1), (2, 1), (5, 1)],
            }
            for c_block, contribs in c_contrib.items():
                for (term_idx, sign) in contribs:
                    if term_idx == b:
                        for ki in range(2):
                            for kj in range(2):
                                c_row = (c_block // 2) * 2 + ki
                                c_col = (c_block % 2) * 2 + kj
                                w_idx = c_col * 4 + c_row
                                local_k = kj
                                local_i = ki
                                W4[w_idx, r] += sign * W2[local_k * 2 + local_i, s]

    total_rank = 49 + 49
    U = np.zeros((25, total_rank), dtype=np.int8)
    V = np.zeros((25, total_rank), dtype=np.int8)
    W = np.zeros((25, total_rank), dtype=np.int8)
    
    U[0:16, 0:49] = U4
    V[0:16, 0:49] = V4
    W[0:16, 0:49] = W4
    
    col = 49
    for i in range(4):
        for k in range(4):
            U[i*5+4, col] = 1
            V[20+k, col] = 1
            W[k*5+i, col] = 1
            col += 1
    for i in range(4):
        for j in range(4):
            U[i*5+j, col] = 1
            V[j*5+4, col] = 1
            W[4*5+i, col] = 1
            col += 1
    for i in range(4):
        U[i*5+4, col] = 1
        V[24, col] = 1
        W[4*5+i, col] = 1
        col += 1
    for k in range(4):
        for j in range(4):
            U[20+j, col] = 1
            V[j*5+k, col] = 1
            W[k*5+4, col] = 1
            col += 1
    for k in range(4):
        U[24, col] = 1
        V[20+k, col] = 1
        W[k*5+4, col] = 1
        col += 1
    for j in range(4):
        U[20+j, col] = 1
        V[j*5+4, col] = 1
        W[24, col] = 1
        col += 1
    U[24, col] = 1
    V[24, col] = 1
    W[24, col] = 1
    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook(n, m, p):
    """Schoolbook decomposition with rank n*m*p."""
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # 1. Try to beat the Strassen-based rank (98) using ALS with lower ranks.
    # Known lower bounds are around 80-90 for 5x5? 
    # Strassen 4x4 is 49. 5x5 is significantly harder.
    # Let's try ranks: 90, 85, 80, 75.
    
    # We will attempt ALS for a few ranks. If we find a valid decomposition 
    # with rank < 98, we return it.
    
    # Time budget per rank attempt
    ranks_to_try = [95, 90, 85, 80]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 20:
            break
            
        # Allocate remaining time / number of remaining ranks
        time_left = deadline - time.time()
        time_per_attempt = time_left / (len(ranks_to_try) - ranks_to_try.index(rank))
        time_per_attempt = min(time_per_attempt, 30) # Cap at 30s per attempt
        
        try:
            result = _try_als_decomposition(rank, time_per_attempt, num_samples=50, seed=rank)
            if result is not None:
                # Verify one more time with more samples to be safe
                if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p, num_tests=10):
                    return result
        except Exception:
            pass
    
    # 2. Fallback to Strassen-based decomposition (rank 98)
    if time.time() < deadline - 5:
        try:
            result = _build_strassen_5x5()
            if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p, num_tests=5):
                return result
        except Exception:
            pass
            
    # 3. Fallback to Schoolbook (rank 125)
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
