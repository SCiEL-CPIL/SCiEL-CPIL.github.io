"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

import time
import sys

# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a rank-85 decomposition for M_{4,5,5}.
    Decomposition:
    1. M_{4,4,4} part (Rank 49 via Strassen)
    2. M_{4,4,1} part (Rank 16)
    3. M_{4,1,4} part (Rank 16)
    4. M_{4,1,1} part (Rank 4)
    Total Rank: 49 + 16 + 16 + 4 = 85.
    """
    n, m, p = 4, 5, 5
    U = [[0] * 85 for _ in range(n * m)] # 20 x 85
    V = [[0] * 85 for _ in range(m * p)] # 25 x 85
    W = [[0] * 85 for _ in range(p * n)] # 20 x 85
    
    # Indices:
    # U: i * 5 + j (i:0-3, j:0-4)
    # V: j * 5 + k (j:0-4, k:0-4)
    # W: k * 4 + i (k:0-4, i:0-3) -- Note: W[k*n+i, r]
    
    # --- Part 1: M_{4,4,4} (A[0:4, 0:4] * B[0:4, 0:4] -> C[0:4, 0:4]) ---
    # Strassen for 4x4.
    # We need U, V, W for M_{2,2,2} first.
    
    # M_{2,2,2} Strassen
    # U (4x7), V (4x7), W (4x7)
    # Input A: A00, A01, A10, A11. Index: 0, 1, 2, 3
    # Input B: B00, B01, B10, B11. Index: 0, 1, 2, 3
    # Output C: C00, C01, C10, C11. Index: 0, 1, 2, 3
    
    U_222 = [[0]*7 for _ in range(4)]
    V_222 = [[0]*7 for _ in range(4)]
    W_222 = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Col 0: M1
    U_222[0][0] = 1 # A00
    U_222[3][0] = 1 # A11
    V_222[0][0] = 1 # B00
    V_222[3][0] = 1 # B11
    # C00 += M1, C11 += M1
    W_222[0][0] = 1
    W_222[3][0] = 1
    
    # Col 1: M2
    U_222[2][1] = 1 # A10
    U_222[3][1] = 1 # A11
    V_222[0][1] = 1 # B00
    # C10 += M2
    W_222[2][1] = 1
    
    # Col 2: M3
    U_222[0][2] = 1 # A00
    V_222[1][2] = 1 # B01
    V_222[3][2] = -1 # B11
    # C01 += M3
    W_222[1][2] = 1
    
    # Col 3: M4
    U_222[3][3] = 1 # A11
    V_222[2][3] = 1 # B10
    V_222[0][3] = -1 # B00
    # C00 += M4, C10 += M4
    W_222[0][3] = 1
    W_222[2][3] = 1
    
    # Col 4: M5
    U_222[0][4] = 1 # A00
    U_222[1][4] = 1 # A01
    V_222[3][4] = 1 # B11
    # C00 -= M5, C01 += M5
    W_222[0][4] = -1
    W_222[1][4] = 1
    
    # Col 5: M6
    U_222[2][5] = 1 # A10
    U_222[0][5] = -1 # A00
    V_222[0][5] = 1 # B00
    V_222[1][5] = 1 # B01
    # C11 += M6
    W_222[3][5] = 1
    
    # Col 6: M7
    U_222[1][6] = 1 # A01
    U_222[3][6] = -1 # A11
    V_222[2][6] = 1 # B10
    V_222[3][6] = 1 # B11
    # C00 += M7
    W_222[0][6] = 1
    
    # Now build M_{4,4,4} using 7 copies of M_{2,2,2}
    # Global A is 4x4. Blocks A00, A01, A10, A11 (each 2x2)
    # Global B is 4x4. Blocks B00, B01, B10, B11 (each 2x2)
    # Global C is 4x4. Blocks C00, C01, C10, C11 (each 2x2)
    
    # Strassen steps for 4x4:
    # S1: A_S1 = A00+A11, B_S1 = B00+B11 -> Result R1 (2x2)
    # S2: A_S2 = A10+A11, B_S2 = B00 -> Result R2
    # S3: A_S3 = A00, B_S3 = B01-B11 -> Result R3
    # S4: A_S4 = A11, B_S4 = B10-B00 -> Result R4
    # S5: A_S5 = A00+A01, B_S5 = B11 -> Result R5
    # S6: A_S6 = A10-A00, B_S6 = B00+B01 -> Result R6
    # S7: A_S7 = A01-A11, B_S7 = B10+B11 -> Result R7
    
    # C00 = R1 + R4 - R5 + R7
    # C01 = R3 + R5
    # C10 = R2 + R4
    # C11 = R1 - R2 + R3 + R6
    
    # For each step s, we have a 2x2 matrix A_s and B_s.
    # We use U_222, V_222, W_222 to decompose A_s * B_s = R_s.
    # The linear forms for A_s and B_s on the global A, B blocks determine U, V.
    # The linear forms for C_s (which is sum of R_s with signs) on global C determine W.
    
    # Helper: get index of global A[i,j]
    def u_idx(i, j): return i * 5 + j
    def v_idx(j, k): return j * 5 + k
    def w_idx(k, i): return k * 4 + i
    
    # Define the linear combinations for each step
    # A_s blocks: list of (block_idx, coeff) where block_idx 0..3 maps to (00,01,10,11)
    # Block 00: rows 0-1, cols 0-1
    # Block 01: rows 0-1, cols 2-3
    # Block 10: rows 2-3, cols 0-1
    # Block 11: rows 2-3, cols 2-3
    
    steps = [
        # s, A_coeffs (dict block->coeff), B_coeffs, C_signs (dict block->sign)
        (1, {0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}), # R1: C00+, C11+
        (2, {2: 1, 3: 1}, {0: 1}, {2: 1}),              # R2: C10+
        (3, {0: 1}, {1: 1, 3: -1}, {1: 1}),             # R3: C01+
        (4, {3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),       # R4: C00+, C10+
        (5, {0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),       # R5: C00-, C01+
        (6, {2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),       # R6: C11+
        (7, {1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})        # R7: C00+
    ]
    
    col = 0
    for s_id, a_coeffs, b_coeffs, c_signs in steps:
        # Iterate through the 7 rank-1 components of M_{2,2,2}
        for k in range(7):
            # Add 7 columns for this step
            # U: For each element (lr, lc) in local A (2x2), we have U_222[lr*2+lc, k]
            # This element corresponds to a linear combination of global A blocks.
            for lr in range(2):
                for lc in range(2):
                    u_val = U_222[lr*2+lc][k]
                    if u_val == 0: continue
                    # Local A[lr,lc] is formed by sum of global A blocks
                    # Global A block index: block_idx = lr*2 + lc
                    # The coefficient of global A_block in Local A[lr,lc] is a_coeffs.get(block_idx, 0)
                    # Wait. A_s is a 2x2 matrix.
                    # A_s[0,0] corresponds to the top-left element of the 2x2 matrix A_s.
                    # A_s is constructed from global blocks.
                    # Example: Step 1: A_S1 = A00 + A11.
                    # A_S1 is a 2x2 matrix.
                    # A_S1[0,0] = A00[0,0] + A11[0,0].
                    # So for local index (0,0), the global block is 0 (A00) with coeff 1, and 3 (A11) with coeff 1.
                    
                    # The linear form for A_s[lr,lc] on global A is:
                    # Sum_{b in a_coeffs} a_coeffs[b] * A_block_b[lr,lc]
                    # So for each global block b in a_coeffs:
                    for b_idx, coeff in a_coeffs.items():
                        if coeff == 0: continue
                        # Global block b_idx has rows 2*(b_idx//2) to 2*(b_idx//2)+1
                        # and cols 2*(b_idx%2) to 2*(b_idx%2)+1
                        gi_start = 2 * (b_idx // 2)
                        gj_start = 2 * (b_idx % 2)
                        gi = gi_start + lr
                        gj = gj_start + lc
                        U[u_idx(gi, gj)][col] += u_val * coeff
                        
            # V: Similar for B
            for lr in range(2):
                for lc in range(2):
                    v_val = V_222[lr*2+lc][k]
                    if v_val == 0: continue
                    for b_idx, coeff in b_coeffs.items():
                        if coeff == 0: continue
                        gi_start = 2 * (b_idx // 2)
                        gj_start = 2 * (b_idx % 2)
                        gj = gi_start + lr # Row index of B is j
                        gk = gj_start + lc # Col index of B is k
                        V[v_idx(gj, gk)][col] += v_val * coeff
                        
            # W: For each element (lr, lc) in local C (2x2), we have W_222[lr*2+lc, k]
            # Local C[lr,lc] is formed by sum of global C blocks with signs.
            for lr in range(2):
                for lc in range(2):
                    w_val = W_222[lr*2+lc][k]
                    if w_val == 0: continue
                    # Local C[lr,lc] corresponds to global C block index: b_idx = lr*2 + lc
                    # But the sign depends on the step.
                    # c_signs maps block_idx to sign.
                    b_idx = lr*2 + lc
                    sign = c_signs.get(b_idx, 0)
                    if sign == 0: continue
                    gi_start = 2 * (b_idx // 2)
                    gj_start = 2 * (b_idx % 2)
                    gk = gj_start + lc # Col index of C is k
                    gi = gi_start + lr # Row index of C is i
                    W[w_idx(gk, gi)][col] += w_val * sign
                    
            col += 1
            
    # --- Part 2: M_{4,4,1} ---
    # A[0:4, 0:4] * B[0:4, 4] -> C[0:4, 4]
    # C_i = sum_j A[i,j] B[j,4]
    # 16 terms
    for i in range(4):
        for j in range(4):
            U[u_idx(i, j)][col] = 1
            V[v_idx(j, 4)][col] = 1
            W[w_idx(4, i)][col] = 1
            col += 1
            
    # --- Part 3: M_{4,1,4} ---
    # A[0:4, 4] * B[4, 0:4] -> C[0:4, 0:4]
    # C_i,k = A[i,4] B[4,k]
    # 16 terms
    for i in range(4):
        for k in range(4):
            U[u_idx(i, 4)][col] = 1
            V[v_idx(4, k)][col] = 1
            W[w_idx(k, i)][col] = 1
            col += 1
            
    # --- Part 4: M_{4,1,1} ---
    # A[0:4, 4] * B[4, 4] -> C[0:4, 4]
    # C_i,4 = A[i,4] B[4,4]
    # 4 terms
    for i in range(4):
        U[u_idx(i, 4)][col] = 1
        V[v_idx(4, 4)][col] = 1
        W[w_idx(4, i)][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
