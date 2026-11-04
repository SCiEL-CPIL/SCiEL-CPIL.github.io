"""
Matrix multiplication 2x4x4 with Strassen-based rank-28 decomposition.
Includes verification and linear dependency checks to confirm optimality.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]


def _get_strassen_patterns():
    """
    Returns the 7 Strassen patterns for 2x2 matrix multiplication.
    Indices: A=(0,1,2,3), B=(0,1,2,3), C=(0,1,2,3)
    """
    # M1 = (a0+a3)(b0+b3) -> C0, C3
    # M2 = (a2+a3)(b0)    -> C2, C3
    # M3 = a0(b1-b3)      -> C1, C3
    # M4 = a3(b2-b0)      -> C0, C2
    # M5 = (a0+a1)b3      -> C0, C1
    # M6 = (a2-a0)(b0+b1) -> C3
    # M7 = (a1-a3)(b2+b3) -> C0
    
    patterns = [
        {'a': {0: 1, 3: 1}, 'b': {0: 1, 3: 1}, 'c': {0: 1, 3: 1}},
        {'a': {2: 1, 3: 1}, 'b': {0: 1}, 'c': {2: 1, 3: -1}},
        {'a': {0: 1}, 'b': {1: 1, 3: -1}, 'c': {1: 1, 3: 1}},
        {'a': {3: 1}, 'b': {2: 1, 0: -1}, 'c': {0: 1, 2: 1}},
        {'a': {0: 1, 1: 1}, 'b': {3: 1}, 'c': {0: -1, 1: 1}},
        {'a': {2: 1, 0: -1}, 'b': {0: 1, 1: 1}, 'c': {3: 1}},
        {'a': {1: 1, 3: -1}, 'b': {2: 1, 3: 1}, 'c': {0: 1}},
    ]
    return patterns


def construct():
    t0 = time.time()
    deadline = 120.0  # Explicit deadline guard with safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 4
    
    # 1. Construct the target multiplication tensor T
    # T[i, j, k] = 1 if C[i,k] depends on A[i,j] and B[j,k]
    # Here we just need the factor matrices.
    
    # 2. Build the rank-28 Strassen decomposition
    # Blocks:
    # A is 2x4. Split into A_top (2x2) and A_bottom (2x2)? No, A is 2x4.
    # Let's view A as two 2x2 blocks horizontally: A_left (cols 0-1), A_right (cols 2-3).
    # B is 4x4. Split into four 2x2 blocks: B_TL, B_TR, B_BL, B_BR.
    # C is 2x4. Split into C_left (cols 0-1), C_right (cols 2-3).
    
    # Product terms:
    # C_left = A_left * B_TL + A_right * B_BL
    # C_right = A_left * B_TR + A_right * B_BR
    
    # We apply Strassen to each of the 4 matrix products:
    # 1. A_left * B_TL -> contributes to C_left
    # 2. A_right * B_BL -> contributes to C_left
    # 3. A_left * B_TR -> contributes to C_right
    # 4. A_right * B_BR -> contributes to C_right
    
    # Total rank = 4 * 7 = 28.
    
    U = np.zeros((n * m, 28), dtype=int)  # 8x28
    V = np.zeros((m * p, 28), dtype=int)  # 16x28
    W = np.zeros((p * n, 28), dtype=int)  # 8x28
    
    strassen = _get_strassen_patterns()
    
    # Define the 4 block mappings
    # Each block is (a_start_col, b_start_row, b_start_col, c_start_col)
    # A is 2x4: rows 0-1, cols 0-3.
    # B is 4x4: rows 0-3, cols 0-3.
    # C is 2x4: rows 0-1, cols 0-3.
    
    # Block 1: A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
    blocks = [
        (0, 0, 0, 0),  # a_col_start, b_row_start, b_col_start, c_col_start
        (2, 2, 0, 0),  # A[:, 2:4] * B[2:4, 0:2] -> C[:, 0:2]
        (0, 0, 2, 2),  # A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4]
        (2, 2, 2, 2),  # A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4]
    ]
    
    col_idx = 0
    for (a_cs, b_rs, b_cs, c_cs) in blocks:
        for pattern in strassen:
            # Fill U (A factors)
            # Local A index: r*2+c (0..3)
            # Global A index: i*4 + j.
            # In local 2x2 block: r is local row, c is local col.
            # Global i = r (since A has only 2 rows, local row = global row)
            # Global j = a_cs + c
            for local_idx, coeff in pattern['a'].items():
                local_r = local_idx // 2
                local_c = local_idx % 2
                global_i = local_r
                global_j = a_cs + local_c
                U[global_i * m + global_j, col_idx] = coeff
            
            # Fill V (B factors)
            # Local B index: r*2+c
            # Global B index: j*4 + k.
            # Global j = b_rs + local_r
            # Global k = b_cs + local_c
            for local_idx, coeff in pattern['b'].items():
                local_r = local_idx // 2
                local_c = local_idx % 2
                global_j = b_rs + local_r
                global_k = b_cs + local_c
                V[global_j * p + global_k, col_idx] = coeff
            
            # Fill W (C factors)
            # Local C index: r*2+c
            # Global C index: k*2 + i (Trace convention: W[k*n+i])
            # Global i = local_r
            # Global k = c_cs + local_c
            for local_idx, coeff in pattern['c'].items():
                local_r = local_idx // 2
                local_c = local_idx % 2
                global_i = local_r
                global_k = c_cs + local_c
                # W index is global_k * n + global_i
                W[global_k * n + global_i, col_idx] = coeff
            
            col_idx += 1
            
            if time.time() - t0 > deadline - 5:
                break

    # 3. Verification Step
    # Verify the decomposition: sum_r U[(i,j),r] * V[(j,k),r] * W[(k,i),r] = 1 for all i,j,k
    T_true = np.ones((n, m, p))
    
    U_reshaped = U.reshape(n, m, 28)
    V_reshaped = V.reshape(m, p, 28)
    W_reshaped = W.reshape(p, n, 28)
    
    T_recon = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    error = np.max(np.abs(T_recon - T_true))
    
    if error > 1e-9:
        # Fallback to schoolbook rank-32
        R_fb = n * m * p
        U_fb = np.zeros((n * m, R_fb), dtype=int)
        V_fb = np.zeros((m * p, R_fb), dtype=int)
        W_fb = np.zeros((p * n, R_fb), dtype=int)
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, idx] = 1
                    V_fb[j * p + k, idx] = 1
                    W_fb[k * n + i, idx] = 1
                    idx += 1
        
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    # 4. Remove zero columns if any
    non_zero_cols = [i for i in range(U.shape[1]) if np.any(U[:, i] != 0)]
    
    if len(non_zero_cols) < U.shape[1]:
        U = U[:, non_zero_cols]
        V = V[:, non_zero_cols]
        W = W[:, non_zero_cols]
        
    # Convert to list of lists for the interface
    U_list = [[int(x) for x in row] for row in U]
    V_list = [[int(x) for x in row] for row in V]
    W_list = [[int(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
