import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    hard_deadline = start_time + 140  # 10s safety margin before 150s kill
    
    # Build the rank-28 Strassen decomposition
    rank28 = _build_strassen_rank28()
    
    # Try to find a rank-27 decomposition using algebraic reduction
    elapsed = time.time() - start_time
    if elapsed < 100:  # Leave 50s for reduction attempt
        rank27 = _try_reduce_rank(rank28, n, m, p, hard_deadline)
        if rank27 is not None:
            return rank27
    
    return rank28

def _build_strassen_rank28():
    n, m, p = 2, 4, 4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(block_id, A_rows, A_cols, B_rows, B_cols, C_rows, C_cols):
        base_col = block_id * 7
        
        col = base_col + 0
        U[A_rows[0] * m + A_cols[0]][col] += 1
        U[A_rows[1] * m + A_cols[1]][col] += 1
        V[B_rows[0] * p + B_cols[0]][col] += 1
        V[B_rows[1] * p + B_cols[1]][col] += 1
        
        col = base_col + 1
        U[A_rows[1] * m + A_cols[0]][col] += 1
        U[A_rows[1] * m + A_cols[1]][col] += 1
        V[B_rows[0] * p + B_cols[0]][col] += 1
        
        col = base_col + 2
        U[A_rows[0] * m + A_cols[0]][col] += 1
        V[B_rows[0] * p + B_cols[1]][col] += 1
        V[B_rows[1] * p + B_cols[1]][col] -= 1
        
        col = base_col + 3
        U[A_rows[1] * m + A_cols[1]][col] += 1
        V[B_rows[1] * p + B_cols[0]][col] += 1
        V[B_rows[0] * p + B_cols[0]][col] -= 1
        
        col = base_col + 4
        U[A_rows[0] * m + A_cols[0]][col] += 1
        U[A_rows[0] * m + A_cols[1]][col] += 1
        V[B_rows[1] * p + B_cols[1]][col] += 1
        
        col = base_col + 5
        U[A_rows[1] * m + A_cols[0]][col] += 1
        U[A_rows[0] * m + A_cols[0]][col] -= 1
        V[B_rows[0] * p + B_cols[0]][col] += 1
        V[B_rows[0] * p + B_cols[1]][col] += 1
        
        col = base_col + 6
        U[A_rows[0] * m + A_cols[1]][col] += 1
        U[A_rows[1] * m + A_cols[1]][col] -= 1
        V[B_rows[1] * p + B_cols[0]][col] += 1
        V[B_rows[1] * p + B_cols[1]][col] += 1
        
        w_idx_00 = C_cols[0] * n + C_rows[0]
        W[w_idx_00][base_col + 0] += 1
        W[w_idx_00][base_col + 3] += 1
        W[w_idx_00][base_col + 4] -= 1
        W[w_idx_00][base_col + 6] += 1
        
        w_idx_01 = C_cols[1] * n + C_rows[0]
        W[w_idx_01][base_col + 2] += 1
        W[w_idx_01][base_col + 4] += 1
        
        w_idx_10 = C_cols[0] * n + C_rows[1]
        W[w_idx_10][base_col + 1] += 1
        W[w_idx_10][base_col + 3] += 1
        
        w_idx_11 = C_cols[1] * n + C_rows[1]
        W[w_idx_11][base_col + 0] += 1
        W[w_idx_11][base_col + 1] -= 1
        W[w_idx_11][base_col + 2] += 1
        W[w_idx_11][base_col + 5] += 1
    
    add_strassen_block(0, [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])
    add_strassen_block(1, [0, 1], [2, 3], [2, 3], [0, 1], [0, 1], [0, 1])
    add_strassen_block(2, [0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3])
    add_strassen_block(3, [0, 1], [2, 3], [2, 3], [2, 3], [0, 1], [2, 3])
    
    return {"U": U, "V": V, "W": W}

def _try_reduce_rank(rank28, n, m, p, deadline):
    """Try to find a rank-27 decomposition by checking if one term is linearly dependent."""
    U28 = np.array(rank28["U"], dtype=float)
    V28 = np.array(rank28["V"], dtype=float)
    W28 = np.array(rank28["W"], dtype=float)
    
    rank = 28
    num_u = n * m  # 8
    num_v = m * p  # 16
    num_w = p * n  # 8
    
    # Precompute outer products for efficiency
    outers = np.zeros((rank, num_u, num_v))
    for i in range(rank):
        outers[i] = np.outer(U28[:, i], V28[:, i])
    
    # For each candidate rank term to remove, check if the tensor it contributes
    # can be expressed as a combination of the remaining terms.
    for remove_idx in range(rank):
        if time.time() > deadline - 10:
            return None
        
        U_rem = np.delete(U28, remove_idx, axis=1)
        V_rem = np.delete(V28, remove_idx, axis=1)
        
        # Build the linear system matrix: each column is outer(U_rem[:,r], V_rem[:,r]).flatten()
        A = np.zeros((num_u * num_v, rank - 1))
        for r in range(rank - 1):
            A[:, r] = np.outer(U_rem[:, r], V_rem[:, r]).flatten()
        
        # Check if A has full column rank
        rank_A = np.linalg.matrix_rank(A)
        if rank_A < (rank - 1):
            continue
        
        # Precompute the left-hand side for the removed term
        removed_outer_flat = outers[remove_idx].flatten()
        
        all_valid = True
        C_coeffs = np.zeros((num_w, rank - 1))
        
        for k in range(num_w):
            if time.time() > deadline - 5:
                return None
            
            lhs = W28[k, remove_idx] * removed_outer_flat
            
            # Solve A @ coeffs = lhs
            try:
                coeffs, residuals, rank_sol, sv = np.linalg.lstsq(A, lhs, rcond=1e-10)
                residual = np.linalg.norm(A @ coeffs - lhs)
                if residual > 1e-8:
                    all_valid = False
                    break
                C_coeffs[k, :] = coeffs
            except:
                all_valid = False
                break
        
        if all_valid:
            return {
                "U": U_rem.tolist(),
                "V": V_rem.tolist(),
                "W": C_coeffs.tolist()
            }
    
    # Method 2: Try to find rank-27 by checking tensor rank directly
    if time.time() < deadline - 15:
        # Build the full 8x16x8 tensor
        T = np.zeros((num_u, num_v, num_w))
        for i in range(rank):
            T += W28[:, i][:, None, None] * outers[i][:, :, None]
        
        # Check if the tensor has rank <= 27 by trying to find a 
        # 27-term decomposition via SVD-based initialization
        result = _try_numerical_rank27(T, outers, U28, V28, W28, deadline)
        if result is not None:
            return result
    
    return None

def _try_numerical_rank27(T, outers, U28, V28, W28, deadline):
    """Try to find a rank-27 CP decomposition using the Strassen structure."""
    num_u, num_v, num_w = T.shape
    target_rank = 27
    
    # Strategy: try dropping each term and see if we can fit the residual
    # with a modified set of 27 terms
    # Use the fact that the 28 terms come from 4 Strassen blocks of 7
    
    # Try: can we merge two terms from different blocks into one?
    # For each pair of terms (i, j) from different blocks, check if
    # u_i ⊗ v_i ⊗ w_i + u_j ⊗ v_j ⊗ w_j can be replaced by one term
    
    # Simpler: try all single-term removals with a relaxed check
    # allowing the W coefficients to be any real numbers (not just from the system)
    
    # Actually, let's try: for each term to remove, solve the full system
    # but allow the U and V to also be adjusted slightly
    
    # Most practical: check if the 28 rank-1 tensors are linearly dependent
    # by checking the rank of the "coefficient matrix"
    
    # Flatten each rank-1 tensor u_i ⊗ v_i ⊗ w_i into a vector of length 8*16*8=1024
    # Check if these 28 vectors are linearly dependent
    tensor_vecs = np.zeros((num_u * num_v * num_w, 28))
    for i in range(28):
        tensor_vecs[:, i] = (outers[i] * W28[:, i][:, None, None]).flatten()
    
    # Check rank of this 1024 x 28 matrix
    tv_rank = np.linalg.matrix_rank(tensor_vecs)
    
    if tv_rank < 28:
        # The 28 terms are linearly dependent! Find the null space
        _, _, vh = np.linalg.svd(tensor_vecs, full_matrices=False)
        # The last row of vh is the null space vector
        null_vec = vh[-1]
        
        # Find the index with largest absolute value in null_vec
        max_idx = np.argmax(np.abs(null_vec))
        
        # Try removing this term: express it as combination of others
        # sum_i null_vec[i] * tensor_i = 0
        # So: tensor_max_idx = -sum_{i≠max_idx} (null_vec[i]/null_vec[max_idx]) * tensor_i
        # But this gives a linear combination of the full rank-1 tensors,
        # not a valid rank-1 decomposition of the residual
        
        # This only tells us the terms are dependent, not that we can reduce rank
        # We still need the specific structure check from Method 1
        
        pass
    
    return None
