import numpy as np
import time

_DEADLINE = time.time() + 110

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Total rank calculation:
    # 4x4x4 part: 49 (Strassen)
    # 4x4x3 part: 4 * Rank(2x2x3) = 4 * 11 = 44
    # Total: 93
    total_rank = 49 + 44
    
    # Initialize as numpy arrays for easier manipulation
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    current_rank = 0
    
    # --- Part 1: 4x4x4 using Strassen (Rank 49) ---
    # Define Strassen 4x4x4 as 7 blocks of 2x2x2 Strassen
    
    # Outer coefficients for the 7 Strassen terms in 4x4x4
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each entry is (A_block_coeffs, B_block_coeffs)
    # Block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    outer_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}),       # M1
        ({2: 1, 3: 1}, {0: 1}),             # M2
        ({0: 1}, {1: 1, 3: -1}),            # M3
        ({3: 1}, {2: 1, 0: -1}),            # M4
        ({0: 1, 1: 1}, {3: 1}),             # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}),      # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1})       # M7
    ]
    
    # Inner 2x2x2 Strassen terms
    # (A_local_coeffs, B_local_coeffs, C_local_coeffs)
    inner_terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}), # M1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),          # M2
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),          # M3
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),          # M4
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),          # M5
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),          # M6
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})           # M7
    ]
    
    # C block mapping for each outer term
    # c_block_coeffs: {(row_block, col_block): coeff}
    c_block_map = [
        {(0,0): 1.0, (1,1): 1.0},       # M1
        {(1,0): 1.0, (1,1): -1.0},      # M2
        {(0,1): 1.0, (1,1): 1.0},       # M3
        {(0,0): 1.0, (1,0): 1.0},       # M4
        {(0,0): -1.0, (0,1): 1.0},      # M5
        {(1,1): 1.0},                   # M6
        {(0,0): 1.0}                    # M7
    ]

    for m_idx in range(7):
        x_def, y_def = outer_terms[m_idx]
        for t_idx in range(7):
            r = current_rank
            current_rank += 1
            x_sub, y_sub, c_sub = inner_terms[t_idx]
            
            # U coefficients (A)
            for (li, lj), c_x in x_sub.items():
                for b, coeff_block in x_def.items():
                    bi = b // 2
                    bj = b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    u_idx = i * m + j
                    U[u_idx, r] += c_x * coeff_block
            
            # V coefficients (B)
            for (li, lj), c_y in y_sub.items():
                for b, coeff_block in y_def.items():
                    bi = b // 2
                    bj = b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    v_idx = i * p + j
                    V[v_idx, r] += c_y * coeff_block
            
            # W coefficients (C)
            c_blocks = c_block_map[m_idx]
            for (li, lj), cm_c in c_sub.items():
                if cm_c == 0.0: continue
                for (br, bc), cb_c in c_blocks.items():
                    if cb_c == 0.0: continue
                    i_final = 2 * br + li
                    k_final = 2 * bc + lj
                    w_idx = k_final * n + i_final
                    W[w_idx, r] += cm_c * cb_c

    # --- Part 2: 4x4x3 using 4 * (2x2x3) (Rank 44) ---
    # Handles columns 4, 5, 6 of B.
    
    def add_2x2x3(a_row_block, a_col_block, b_row_block, c_row_block):
        nonlocal current_rank
        
        # 11 terms: 7 Strassen (for B cols 4,5) + 4 Schoolbook (for B col 6)
        
        # Strassen part (cols 4,5)
        for t_idx in range(7):
            r = current_rank
            current_rank += 1
            x_sub, y_sub, c_sub = inner_terms[t_idx]
            
            # A entries
            for (li, lj), c_x in x_sub.items():
                i = 2 * a_row_block + li
                j = 2 * a_col_block + lj
                u_idx = i * m + j
                U[u_idx, r] += c_x
            
            # B entries (cols 4, 5)
            for (li, lj), c_y in y_sub.items():
                i = 2 * b_row_block + li
                j = 4 + lj 
                v_idx = i * p + j
                V[v_idx, r] += c_y
            
            # C entries (cols 4, 5)
            for (li, lj), cm_c in c_sub.items():
                if cm_c == 0.0: continue
                i_final = 2 * c_row_block + li
                k_final = 4 + lj 
                w_idx = k_final * n + i_final
                W[w_idx, r] += cm_c
        
        # Schoolbook part (col 6) — 4 terms
        # Term 1: A[0,0] * B[0,6] -> C[0,6]
        r = current_rank
        current_rank += 1
        U[(2 * a_row_block + 0) * m + (2 * a_col_block + 0), r] = 1.0
        V[(2 * b_row_block + 0) * p + 6, r] = 1.0
        W[6 * n + (2 * c_row_block + 0), r] = 1.0

        # Term 2: A[0,1] * B[1,6] -> C[0,6]
        r = current_rank
        current_rank += 1
        U[(2 * a_row_block + 0) * m + (2 * a_col_block + 1), r] = 1.0
        V[(2 * b_row_block + 1) * p + 6, r] = 1.0
        W[6 * n + (2 * c_row_block + 0), r] = 1.0

        # Term 3: A[1,0] * B[0,6] -> C[1,6]
        r = current_rank
        current_rank += 1
        U[(2 * a_row_block + 1) * m + (2 * a_col_block + 0), r] = 1.0
        V[(2 * b_row_block + 0) * p + 6, r] = 1.0
        W[6 * n + (2 * c_row_block + 1), r] = 1.0

        # Term 4: A[1,1] * B[1,6] -> C[1,6]
        r = current_rank
        current_rank += 1
        U[(2 * a_row_block + 1) * m + (2 * a_col_block + 1), r] = 1.0
        V[(2 * b_row_block + 1) * p + 6, r] = 1.0
        W[6 * n + (2 * c_row_block + 1), r] = 1.0

    # Call for the 4 blocks
    # A00 (rows 0-1, cols 0-1) * B_top (rows 0-1, cols 4-6) -> C_top (rows 0-1, cols 4-6)
    add_2x2x3(0, 0, 0, 0)
    
    # A01 (rows 0-1, cols 2-3) * B_bot (rows 2-3, cols 4-6) -> C_top (rows 0-1, cols 4-6)
    add_2x2x3(0, 1, 1, 0)
    
    # A10 (rows 2-3, cols 0-1) * B_top (rows 0-1, cols 4-6) -> C_bot (rows 2-3, cols 4-6)
    add_2x2x3(1, 0, 0, 1)
    
    # A11 (rows 2-3, cols 2-3) * B_bot (rows 2-3, cols 4-6) -> C_bot (rows 2-3, cols 4-6)
    add_2x2x3(1, 1, 1, 1)

    # Trim to actual used rank
    U = U[:, :current_rank]
    V = V[:, :current_rank]
    W = W[:, :current_rank]

    # Verify correctness
    if time.time() < _DEADLINE:
        target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    target[a_idx, b_idx, c_idx] = 1.0
        
        recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        
        max_err = np.max(np.abs(recon - target))
        if max_err > 1e-10:
            # Fallback: return schoolbook (rank 112)
            rank_sb = n * m * p
            U_sb = np.zeros((n*m, rank_sb))
            V_sb = np.zeros((m*p, rank_sb))
            W_sb = np.zeros((p*n, rank_sb))
            r = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_sb[i*m+j, r] = 1.0
                        V_sb[j*p+k, r] = 1.0
                        W_sb[k*n+i, r] = 1.0
                        r += 1
            return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
