import numpy as np
import time

_DEADLINE = time.time() + 110  # Safety margin before 120s limit

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Total rank: 49 (Strassen 4x4x4) + 44 (4 * Rank(2x2x3) = 4 * 11)
    total_rank = 49 + 44
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    current_rank = 0
    
    # --- Part 1: 4x4x4 using Strassen (Rank 49) ---
    # Strassen 4x4x4 decomposition into 7 blocks of 2x2x2
    # Outer terms define how the 4x4 matrices are split into 2x2 blocks
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Mapping for outer terms: {block_index: coefficient}
    # Block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen_terms_4x4 = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}),       # M1
        ({2: 1, 3: 1}, {0: 1}),             # M2
        ({0: 1}, {1: 1, 3: -1}),            # M3
        ({3: 1}, {2: 1, 0: -1}),            # M4
        ({0: 1, 1: 1}, {3: 1}),             # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}),      # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1})       # M7
    ]
    
    # Inner 2x2x2 Strassen terms
    # x_sub, y_sub define the 2x2 sub-matrices for A and B
    # c_sub defines the contribution to the 2x2 C sub-matrix
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_terms_2x2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}), # M1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),          # M2
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),          # M3
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),          # M4
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),          # M5
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),          # M6
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})           # M7
    ]

    for m_idx in range(7):
        x_def, y_def = strassen_terms_4x4[m_idx]
        for t_idx in range(7):
            r = current_rank
            current_rank += 1
            x_sub, y_sub, c_sub = strassen_terms_2x2[t_idx]
            
            u_coeffs = [0.0] * (n*m)
            v_coeffs = [0.0] * (m*p)
            
            # Populate U (A matrix)
            for (li, lj), c_x in x_sub.items():
                for b, coeff_block in x_def.items():
                    bi = b // 2
                    bj = b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    u_idx = i * m + j
                    u_coeffs[u_idx] += c_x * coeff_block
            
            # Populate V (B matrix)
            for (li, lj), c_y in y_sub.items():
                for b, coeff_block in y_def.items():
                    bi = b // 2
                    bj = b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    v_idx = i * p + j
                    v_coeffs[v_idx] += c_y * coeff_block
            
            # Populate W (C matrix)
            w_coeffs = [0.0] * (p*n)
            
            # Determine which 2x2 blocks of C are affected by this outer term m_idx
            # c_block_coeffs maps (row_block, col_block) to coefficient
            c_block_coeffs = {}
            if m_idx == 0: # M1
                c_block_coeffs = {(0,0): 1.0, (1,1): 1.0}
            elif m_idx == 1: # M2
                c_block_coeffs = {(1,0): 1.0, (1,1): -1.0}
            elif m_idx == 2: # M3
                c_block_coeffs = {(0,1): 1.0, (1,1): 1.0}
            elif m_idx == 3: # M4
                c_block_coeffs = {(0,0): 1.0, (1,0): 1.0}
            elif m_idx == 4: # M5
                c_block_coeffs = {(0,0): -1.0, (0,1): 1.0}
            elif m_idx == 5: # M6
                c_block_coeffs = {(1,1): 1.0}
            elif m_idx == 6: # M7
                c_block_coeffs = {(0,0): 1.0}
            
            for (li, lj), cm_c in c_sub.items():
                if cm_c == 0.0: continue
                for (br, bc), cb_c in c_block_coeffs.items():
                    if cb_c == 0.0: continue
                    i_final = 2 * br + li
                    k_final = 2 * bc + lj
                    w_idx = k_final * n + i_final
                    w_coeffs[w_idx] += cm_c * cb_c
            
            for i in range(n*m):
                U[i][r] = u_coeffs[i]
            for j in range(m*p):
                V[j][r] = v_coeffs[j]
            for i in range(p*n):
                W[i][r] = w_coeffs[i]

    # --- Part 2: 4x4x3 using 4 * (2x2x3) (Rank 44) ---
    # Handles columns 4, 5, 6 of B.
    # We split A into 2x2 blocks (A00, A01, A10, A11).
    # We split B (cols 4-6) into top rows (0,1) and bottom rows (2,3).
    # A00 * B_top -> C_top (cols 4-6)
    # A01 * B_bot -> C_top (cols 4-6)
    # A10 * B_top -> C_bot (cols 4-6)
    # A11 * B_bot -> C_bot (cols 4-6)
    
    def add_2x2x3(a_row_block, a_col_block, b_row_block, c_row_block):
        nonlocal current_rank
        
        # a_row_block, a_col_block: 0 or 1 (offset in A)
        # b_row_block: 0 or 1 (offset in B rows)
        # c_row_block: 0 or 1 (offset in C rows)
        
        # 11 terms: 7 Strassen (for B cols 4,5) + 4 Schoolbook (for B col 6)
        
        # Strassen part (cols 4,5)
        for t_idx in range(7):
            r = current_rank
            current_rank += 1
            x_sub, y_sub, c_sub = strassen_terms_2x2[t_idx]
            
            u_coeffs = [0.0] * (n*m)
            v_coeffs = [0.0] * (m*p)
            
            # A entries
            for (li, lj), c_x in x_sub.items():
                i = 2 * a_row_block + li
                j = 2 * a_col_block + lj
                u_idx = i * m + j
                u_coeffs[u_idx] += c_x
            
            # B entries (cols 4, 5)
            for (li, lj), c_y in y_sub.items():
                i = 2 * b_row_block + li
                j = 4 + lj # Col 4 or 5
                v_idx = i * p + j
                v_coeffs[v_idx] += c_y
            
            for i in range(n*m):
                U[i][r] = u_coeffs[i]
            for j in range(m*p):
                V[j][r] = v_coeffs[j]
            
            w_coeffs = [0.0] * (p*n)
            
            # C entries (cols 4, 5)
            for (li, lj), cm_c in c_sub.items():
                if cm_c == 0.0: continue
                i_final = 2 * c_row_block + li
                k_final = 4 + lj # Col 4 or 5
                w_idx = k_final * n + i_final
                w_coeffs[w_idx] += cm_c
            
            for i in range(p*n):
                W[i][r] = w_coeffs[i]
        
        # Schoolbook part (col 6) — 4 terms
        # C[i,6] = A[i,0]B[0,6] + A[i,1]B[1,6]
        # Local A: rows 2ar, 2ar+1; cols 2ac, 2ac+1
        # Local B: rows 2br, 2br+1; col 6
        # Local C: rows 2cr, 2cr+1; col 6

        # Term 1: A[0,0] * B[0,6] -> C[0,6]
        r = current_rank
        current_rank += 1
        U[(2 * a_row_block + 0) * m + (2 * a_col_block + 0)][r] = 1.0
        V[(2 * b_row_block + 0) * p + 6][r] = 1.0
        W[6 * n + (2 * c_row_block + 0)][r] = 1.0

        # Term 2: A[0,1] * B[1,6] -> C[0,6]
        r = current_rank
        current_rank += 1
        U[(2 * a_row_block + 0) * m + (2 * a_col_block + 1)][r] = 1.0
        V[(2 * b_row_block + 1) * p + 6][r] = 1.0
        W[6 * n + (2 * c_row_block + 0)][r] = 1.0

        # Term 3: A[1,0] * B[0,6] -> C[1,6]
        r = current_rank
        current_rank += 1
        U[(2 * a_row_block + 1) * m + (2 * a_col_block + 0)][r] = 1.0
        V[(2 * b_row_block + 0) * p + 6][r] = 1.0
        W[6 * n + (2 * c_row_block + 1)][r] = 1.0

        # Term 4: A[1,1] * B[1,6] -> C[1,6]
        r = current_rank
        current_rank += 1
        U[(2 * a_row_block + 1) * m + (2 * a_col_block + 1)][r] = 1.0
        V[(2 * b_row_block + 1) * p + 6][r] = 1.0
        W[6 * n + (2 * c_row_block + 1)][r] = 1.0

    # Call for the 4 blocks
    # A00 (rows 0-1, cols 0-1) * B_top (rows 0-1, cols 4-6) -> C_top (rows 0-1, cols 4-6)
    add_2x2x3(0, 0, 0, 0)
    
    # A01 (rows 0-1, cols 2-3) * B_bot (rows 2-3, cols 4-6) -> C_top (rows 0-1, cols 4-6)
    add_2x2x3(0, 1, 1, 0)
    
    # A10 (rows 2-3, cols 0-1) * B_top (rows 0-1, cols 4-6) -> C_bot (rows 2-3, cols 4-6)
    add_2x2x3(1, 0, 0, 1)
    
    # A11 (rows 2-3, cols 2-3) * B_bot (rows 2-3, cols 4-6) -> C_bot (rows 2-3, cols 4-6)
    add_2x2x3(1, 1, 1, 1)

    # Trim
    U = [row[:current_rank] for row in U]
    V = [row[:current_rank] for row in V]
    W = [row[:current_rank] for row in W]

    # Verify correctness (optional but good for safety)
    if time.time() < _DEADLINE:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Reconstruct tensor: T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
        # Target: T[i,j,k] = 1 if C[i,k] = sum_j A[i,j]*B[j,k]
        # Indices:
        # A: i*m + j
        # B: j*p + k
        # C: k*n + i
        
        target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    target[a_idx, b_idx, c_idx] = 1.0
        
        recon = np.einsum('ir,jr,kr->ijk', U_np, V_np, W_np)
        
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
    
    return {"U": U, "V": V, "W": W}
