import time
import random
import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    n_rows = n * m
    m_rows = m * p
    p_rows = p * n
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n_rows)]
        V = [[0] * rank for _ in range(m_rows)]
        W = [[0] * rank for _ in range(p_rows)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Try to find a rank-9 decomposition for 2x3x3
    # Then use it for 2x3x9 by repeating 3 times
    
    n_small, m_small, p_small = 2, 3, 3
    rank_small = 9
    
    U_small = np.zeros((n_small * m_small, rank_small), dtype=np.int64)
    V_small = np.zeros((m_small * p_small, rank_small), dtype=np.int64)
    W_small = np.zeros((p_small * n_small, rank_small), dtype=np.int64)
    
    found = False
    max_trials = 5000
    for t in range(max_trials):
        if time.time() > deadline - 20:
            break
            
        # Random initialization
        U_small = np.random.randint(-2, 3, size=(6, rank_small))
        V_small = np.random.randint(-2, 3, size=(9, rank_small))
        W_small = np.random.randint(-2, 3, size=(6, rank_small))
        
        # Check if this decomposition is valid
        # T[i, j, k] = sum_c U[i, c] * V[j, c] * W[k, c]
        # We need T[i, j, k] = 1 for all i, j, k?
        # No, the tensor T for matrix multiplication is not all 1s.
        # T[i, j, k] is the coefficient of A[i] * B[j] in C[k].
        # For matrix multiplication, C[i, k] = sum_j A[i, j] * B[j, k].
        # So the tensor is defined on indices (i, j, k) where i is row of A, j is col of A / row of B, k is col of B.
        # But the flattened indices are:
        # A index: i * m + j
        # B index: j * p + k
        # C index: k * n + i
        # The tensor T is such that <T, A \otimes B> = sum_{i, k} C[i, k].
        # This implies T[(i*m+j), (j*p+k), (k*n+i)] = 1.
        # All other entries are 0.
        
        valid = True
        for i in range(n_small):
            for j in range(m_small):
                for k in range(p_small):
                    a_idx = i * m_small + j
                    b_idx = j * p_small + k
                    c_idx = k * n_small + i
                    val = 0
                    for c in range(rank_small):
                        val += U_small[a_idx, c] * V_small[b_idx, c] * W_small[c_idx, c]
                    if val != 1:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if valid:
            # Check other entries are 0
            for a_idx in range(6):
                for b_idx in range(9):
                    # Determine if this pair (a_idx, b_idx) corresponds to a valid (i, j, k)
                    # a_idx = i * 3 + j
                    # b_idx = j * 3 + k
                    # They match if a_idx % 3 == b_idx // 3
                    if a_idx % 3 != b_idx // 3:
                        val = 0
                        # We need to sum over c: U[a_idx, c] * V[b_idx, c] * W[c_idx, c]
                        # But W depends on k and i.
                        # If a_idx and b_idx don't match, there is no single (i, k) that makes sense?
                        # Actually, the tensor is sparse.
                        # T[a, b, c] is non-zero only if a, b, c are consistent.
                        # Consistency: a = i*m+j, b = j*p+k, c = k*n+i.
                        # So for a given a, b, if they are consistent, c is determined.
                        # If they are not consistent, T[a, b, c] must be 0 for all c.
                        i_a = a_idx // m_small
                        j_a = a_idx % m_small
                        j_b = b_idx // p_small
                        k_b = b_idx % p_small
                        if j_a != j_b:
                            # Inconsistent
                            for c_idx in range(6):
                                val = 0
                                for c in range(rank_small):
                                    val += U_small[a_idx, c] * V_small[b_idx, c] * W_small[c_idx, c]
                                if val != 0:
                                    valid = False
                                    break
                            if not valid:
                                break
                    else:
                        # Consistent, c_idx is determined
                        i = i_a
                        k = k_b
                        c_idx = k * n_small + i
                        val = 0
                        for c in range(rank_small):
                            val += U_small[a_idx, c] * V_small[b_idx, c] * W_small[c_idx, c]
                        if val != 1:
                            valid = False
                            break
                if not valid:
                    break
            if valid:
                found = True
                break

    if found:
        # Construct the 2x3x9 decomposition
        rank_full = rank_small * 3
        U_full = np.zeros((n_rows, rank_full), dtype=np.int64)
        V_full = np.zeros((m_rows, rank_full), dtype=np.int64)
        W_full = np.zeros((p_rows, rank_full), dtype=np.int64)
        
        for block in range(3):
            # U is the same for all blocks
            U_full[:, block * rank_small : (block + 1) * rank_small] = U_small
            
            # V is different for each block
            # B is 3x9. Block b is columns 3*b to 3*b+2.
            # V index for block b is j * 9 + k.
            # k is 0, 1, 2 within the block.
            # So V[j * 9 + 3*b + k, c] = V_small[j * 3 + k, c]
            for j in range(3):
                for k in range(3):
                    v_idx_full = j * 9 + 3 * block + k
                    v_idx_small = j * 3 + k
                    V_full[v_idx_full, block * rank_small : (block + 1) * rank_small] = V_small[v_idx_small, :]
                    
            # W is different for each block
            # C is 2x9. Block b is columns 3*b to 3*b+2.
            # W index for block b is k * 2 + i.
            # k is 0, 1, 2 within the block.
            # So W[k * 2 + i, c] = W_small[k * 2 + i, c]
            # Wait, W index in small is k * n + i = k * 2 + i.
            # In full, W index is k * n + i.
            # But k in full is 0..8.
            # For block b, k_full = 3 * block + k_small.
            # W index in full is k_full * n + i.
            # So W_full[k_full * 2 + i, c] = W_small[k_small * 2 + i, c]
            for k_small in range(3):
                for i in range(2):
                    k_full = 3 * block + k_small
                    w_idx_full = k_full * 2 + i
                    w_idx_small = k_small * 2 + i
                    W_full[w_idx_full, block * rank_small : (block + 1) * rank_small] = W_small[w_idx_small, :]
        
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    else:
        return schoolbook()
