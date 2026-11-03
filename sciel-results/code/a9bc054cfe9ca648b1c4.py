import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=4
    # Rank = 28 (Strassen-based block decomposition)
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B01 - B11)
    # M4 = A11 (B10 - B00)
    # M5 = (A00 + A01) B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Output C:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Local indices:
    # A: 00:0, 01:1, 10:2, 11:3
    # B: 00:0, 01:1, 10:2, 11:3
    # C: 00:0, 01:1, 10:2, 11:3
    
    strassen_steps = [
        # M1
        {
            'u': {0: 1, 3: 1},
            'v': {0: 1, 3: 1},
            'w': {0: 1, 3: 1}
        },
        # M2
        {
            'u': {2: 1, 3: 1},
            'v': {0: 1},
            'w': {2: 1, 3: -1}
        },
        # M3
        {
            'u': {0: 1},
            'v': {1: 1, 3: -1},
            'w': {1: 1, 3: 1}
        },
        # M4
        {
            'u': {3: 1},
            'v': {2: 1, 0: -1},
            'w': {0: 1, 2: 1}
        },
        # M5
        {
            'u': {0: 1, 1: 1},
            'v': {3: 1},
            'w': {0: -1, 1: 1}
        },
        # M6
        {
            'u': {2: 1, 0: -1},
            'v': {0: 1, 1: 1},
            'w': {3: 1}
        },
        # M7
        {
            'u': {1: 1, 3: -1},
            'v': {2: 1, 3: 1},
            'w': {0: 1}
        }
    ]
    
    # Helper to add a block
    def add_block(a_col_start, b_row_start, b_col_start, c_col_start, col_offset):
        for s in range(7):
            step = strassen_steps[s]
            col = col_offset + s
            
            # Update U
            for local_idx, coeff in step['u'].items():
                l_row = local_idx // 2
                l_col = local_idx % 2
                global_i = l_row
                global_j = a_col_start + l_col
                global_idx = global_i * m + global_j
                U[global_idx][col] += coeff
                
            # Update V
            for local_idx, coeff in step['v'].items():
                l_row = local_idx // 2
                l_col = local_idx % 2
                global_i = b_row_start + l_row
                global_j = b_col_start + l_col
                global_idx = global_i * p + global_j
                V[global_idx][col] += coeff
                
            # Update W
            for local_idx, coeff in step['w'].items():
                l_row = local_idx // 2
                l_col = local_idx % 2
                # W index convention: k*n + i
                # k is col of C, i is row of C
                global_k = c_col_start + l_col
                global_i = l_row
                global_idx = global_k * n + global_i
                W[global_idx][col] += coeff

    # Block 1: A_L (cols 0,1), B_TL (rows 0,1 cols 0,1) -> C_L (cols 0,1)
    add_block(a_col_start=0, b_row_start=0, b_col_start=0, c_col_start=0, col_offset=0)
    
    # Block 2: A_L (cols 0,1), B_TR (rows 0,1 cols 2,3) -> C_R (cols 2,3)
    add_block(a_col_start=0, b_row_start=0, b_col_start=2, c_col_start=2, col_offset=7)
    
    # Block 3: A_R (cols 2,3), B_BL (rows 2,3 cols 0,1) -> C_L (cols 0,1)
    add_block(a_col_start=2, b_row_start=2, b_col_start=0, c_col_start=0, col_offset=14)
    
    # Block 4: A_R (cols 2,3), B_BR (rows 2,3 cols 2,3) -> C_R (cols 2,3)
    add_block(a_col_start=2, b_row_start=2, b_col_start=2, c_col_start=2, col_offset=21)

    # Verify correctness: check that the decomposition is exact
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build the tensor from the decomposition
    # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # where i indexes A (n*m=8), j indexes B (m*p=16), k indexes C (p*n=8)
    
    # The multiplication tensor has T[i,j,k] = 1 if A[i] and B[j] contribute to C[k]
    # A[i] where i = a_row*m + a_col, so a_row = i//m, a_col = i%m
    # B[j] where j = b_row*p + b_col, so b_row = j//p, b_col = j%p
    # C[k] where k = c_col*n + c_row, so c_col = k//n, c_row = k%n
    # T[i,j,k] = 1 if a_row == c_row and a_col == b_row and b_col == c_col
    
    # Check: for each (i,j,k), T[i,j,k] should equal sum_r U[i,r]*V[j,r]*W[k,r]
    # Build expected tensor
    T_expected = np.zeros((n*m, m*p, p*n))
    for a_row in range(n):
        for a_col in range(m):
            for b_row in range(m):
                for b_col in range(p):
                    if a_col == b_row:  # A[a_row, a_col] * B[b_row, b_col] contributes to C[a_row, b_col]
                        i = a_row * m + a_col
                        j = b_row * p + b_col
                        c_row = a_row
                        c_col = b_col
                        k = c_col * n + c_row
                        T_expected[i, j, k] = 1.0
    
    # Compute T from decomposition
    T_decomp = np.einsum('ir,jr,kr->ijk', U_np, V_np, W_np)
    
    # Check if they match
    max_err = np.max(np.abs(T_decomp - T_expected))
    if max_err > 1e-10:
        # Fallback: use naive decomposition
        rank_naive = n * m * p
        U_naive = [[0] * rank_naive for _ in range(n * m)]
        V_naive = [[0] * rank_naive for _ in range(m * p)]
        W_naive = [[0] * rank_naive for _ in range(p * n)]
        
        r = 0
        for a_row in range(n):
            for a_col in range(m):
                for b_row in range(m):
                    for b_col in range(p):
                        if a_col == b_row:
                            i = a_row * m + a_col
                            j = b_row * p + b_col
                            c_row = a_row
                            c_col = b_col
                            k = c_col * n + c_row
                            U_naive[i][r] = 1
                            V_naive[j][r] = 1
                            W_naive[k][r] = 1
                            r += 1
        
        return {"U": U_naive, "V": V_naive, "W": W_naive}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
