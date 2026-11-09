import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank will be 49 (for 4x4x4 part) + 44 (for 4x4x3 part) = 93
    rank = 93
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term to U, V, W
    # u_idx: index in U (0 to n*m-1)
    # v_idx: index in V (0 to m*p-1)
    # w_idx: index in W (0 to p*n-1)
    # col: column in the factor matrices
    def add_term(u_idx, v_idx, w_idx, col, val_u=1, val_v=1, val_w=1):
        U[u_idx][col] += val_u
        V[v_idx][col] += val_v
        W[w_idx][col] += val_w

    # --- Part 1: 4x4x4 multiplication (Strassen) ---
    # We treat A as 2x2 blocks of 2x2, B as 2x2 blocks of 2x2.
    # There are 4 block multiplications.
    # Block (bi, bj) of A multiplies Block (bj, bk) of B?
    # C_ik = sum_j A_ij B_jk.
    # A is 4x4. Blocks A_{00}, A_{01}, A_{10}, A_{11}.
    # B is 4x4. Blocks B_{00}, B_{01}, B_{10}, B_{11}.
    # C_{00} = A_{00}B_{00} + A_{01}B_{10}
    # C_{01} = A_{00}B_{01} + A_{01}B_{11}
    # C_{10} = A_{10}B_{00} + A_{11}B_{10}
    # C_{11} = A_{10}B_{01} + A_{11}B_{11}
    
    # We will iterate over the 4 terms: (A_block, B_block, C_block)
    # Terms:
    # 1. A00 * B00 -> C00
    # 2. A01 * B10 -> C00
    # 3. A00 * B01 -> C01
    # 4. A01 * B11 -> C01
    # 5. A10 * B00 -> C10
    # 6. A11 * B10 -> C10
    # 7. A10 * B01 -> C11
    # 8. A11 * B11 -> C11
    
    # For each term, we use Strassen (7 mults).
    # Total 8 * 7 = 56? No.
    # Strassen computes 2x2 * 2x2.
    # We have 4 distinct 2x2 * 2x2 multiplications?
    # No, we have 8 terms, but they group into 4 Strassen computations?
    # No, each A_block * B_block is a separate 2x2 * 2x2 multiplication.
    # So we have 8 separate 2x2 * 2x2 multiplications.
    # Rank 8 * 7 = 56.
    # But we can combine them?
    # C00 = A00 B00 + A01 B10.
    # This is not a single Strassen multiplication.
    # So we must do 8 separate Strassen multiplications.
    # Rank 56.
    # Wait, earlier I said 49.
    # How do we get 49?
    # We use Strassen on the 4x4x4 tensor directly.
    # This means we don't just do 8 independent 2x2x2 mults.
    # We use the Kronecker product structure of Strassen.
    # $M_{4,4,4} = M_{2,2,2} \otimes M_{2,2,2}$.
    # Rank $7 \times 7 = 49$.
    # This requires the specific Kronecker product construction.
    
    # Let's implement the Kronecker product of Strassen factors.
    # Strassen factors for 2x2x2:
    # U_S (2x2, 7), V_S (2x2, 7), W_S (2x2, 7)
    
    # Indices for 2x2x2:
    # U_S[i, c] for i in 0..3 (A indices)
    # V_S[j, c] for j in 0..3 (B indices)
    # W_S[k, c] for k in 0..3 (C indices)
    
    # Strassen definitions (standard):
    # M1: A11+A22, B22, C11 += M1*B1 ...
    # Let's use the explicit coefficients.
    
    # We will build U_S, V_S, W_S for 2x2x2.
    U_S = [[0]*7 for _ in range(4)]
    V_S = [[0]*7 for _ in range(4)]
    W_S = [[0]*7 for _ in range(4)]
    
    # c=0: M1 = A00+A11, B11, C00 += M1*B0? No.
    # Let's use the standard Strassen formulas:
    # M1 = (A00+A11) * B11 -> C00 += M1, C11 -= M1
    # M2 = A11 * (B00+B11) -> C11 += M2
    # M3 = (A00+A01) * B00 -> C00 += M3, C01 -= M3
    # M4 = (A10-A11) * (B11-B00) -> C10 += M4, C11 -= M4
    # M5 = (A00+A10) * B01 -> C01 += M5
    # M6 = (A01-A10) * B10 -> C10 += M6
    # M7 = (A01+A11) * (B00+B10) -> C01 += M7, C11 -= M7
    
    # Note: Indices are 0-based.
    # A00=0, A01=1, A10=2, A11=3
    # B00=0, B01=1, B10=2, B11=3
    # C00=0, C01=1, C10=2, C11=3
    
    # c=0
    U_S[0][0] = 1; U_S[3][0] = 1
    V_S[3][0] = 1
    W_S[0][0] = 1; W_S[3][0] = -1
    
    # c=1
    U_S[3][1] = 1
    V_S[0][1] = 1; V_S[3][1] = 1
    W_S[3][1] = 1
    
    # c=2
    U_S[0][2] = 1; U_S[1][2] = 1
    V_S[0][2] = 1
    W_S[0][2] = 1; W_S[1][2] = -1
    
    # c=3
    U_S[2][3] = 1; U_S[3][3] = -1
    V_S[3][3] = 1; V_S[0][3] = -1
    W_S[2][3] = 1; W_S[3][3] = -1
    
    # c=4
    U_S[0][4] = 1; U_S[2][4] = 1
    V_S[1][4] = 1
    W_S[1][4] = 1
    
    # c=5
    U_S[1][5] = 1; U_S[2][5] = -1
    V_S[2][5] = 1
    W_S[2][5] = 1
    
    # c=6
    U_S[1][6] = 1; U_S[3][6] = 1
    V_S[0][6] = 1; V_S[2][6] = 1
    W_S[1][6] = 1; W_S[3][6] = -1
    
    # Now Kronecker product for 4x4x4.
    # U_44[i1*2+i2, c1*7+c2] = U_S[i1][c1] * U_S[i2][c2]
    # V_44[j1*2+j2, c1*7+c2] = V_S[j1][c1] * V_S[j2][c2]
    # W_44[k1*2+k2, c1*7+c2] = W_S[k1][c1] * W_S[k2][c2]
    
    col_offset = 0
    for c1 in range(7):
        for c2 in range(7):
            col = col_offset + c1*7 + c2
            for i1 in range(2):
                for i2 in range(2):
                    i = i1*2 + i2
                    val_u = U_S[i1][c1] * U_S[i2][c2]
                    if val_u != 0:
                        U[i][col] += val_u
            for j1 in range(2):
                for j2 in range(2):
                    j = j1*2 + j2
                    val_v = V_S[j1][c1] * V_S[j2][c2]
                    if val_v != 0:
                        V[j][col] += val_v
            for k1 in range(2):
                for k2 in range(2):
                    k = k1*2 + k2
                    val_w = W_S[k1][c1] * W_S[k2][c2]
                    if val_w != 0:
                        W[k][col] += val_w
    col_offset = 49
    
    # --- Part 2: 4x4x3 multiplication ---
    # Rank 44.
    # Split into 4 blocks of 2x2x3.
    # Each 2x2x3 has rank 11.
    # Total 44.
    
    # Helper for 2x2x3 mixed (Strassen for first 2 cols, Schoolbook for 3rd)
    # Strassen part (cols 0,1): Rank 7.
    # Schoolbook part (col 2): Rank 4.
    
    # We need to map the 2x2x3 indices to the global 4x4x3 indices.
    # A blocks: A00, A01, A10, A11.
    # B blocks: B00, B01, B10, B11. (Each 2x3)
    # C blocks: C00, C01, C10, C11. (Each 2x3)
    
    # Terms:
    # 1. A00 * B00 -> C00
    # 2. A01 * B10 -> C00
    # 3. A00 * B01 -> C01
    # 4. A01 * B11 -> C01
    # 5. A10 * B00 -> C10
    # 6. A11 * B10 -> C10
    # 7. A10 * B01 -> C11
    # 8. A11 * B11 -> C11
    
    # For each term, we add 11 columns.
    
    # Define Strassen factors for 2x2x2 (cols 0,1 of B)
    # We can reuse U_S, V_S, W_S but only for cols 0,1 of B?
    # No, Strassen uses all 4 B inputs.
    # In 2x2x3, B has 6 inputs.
    # Strassen part uses B00, B01, B10, B11.
    # Schoolbook part uses B02, B12.
    
    # So for the Strassen part, we use the standard 2x2x2 Strassen.
    # For the Schoolbook part, we use 4 terms:
    # A00*B02, A01*B12, A10*B02, A11*B12.
    
    # Let's iterate over the 8 terms.
    terms = [
        (0, 0, 0), # A00, B00, C00
        (1, 2, 0), # A01, B10, C00  (B10 is rows 2,3 of B? No, B10 is block (1,0))
        # Wait, B is 4x3. Blocks are 2x3.
        # B00 is rows 0,1 cols 0,1,2.
        # B10 is rows 2,3 cols 0,1,2.
        # So B00 and B10 are the top and bottom halves of B.
        # A00 is rows 0,1 cols 0,1.
        # A01 is rows 0,1 cols 2,3.
        
        # C00 is rows 0,1 cols 0,1,2.
        # C01 is rows 0,1 cols 2,3,4? No, C is 4x3.
        # C00 is rows 0,1 cols 0,1,2.
        # C10 is rows 2,3 cols 0,1,2.
        
        # So we only have 2 C blocks: C00 and C10.
        # C00 = A00 B00 + A01 B10.
        # C10 = A10 B00 + A11 B10.
        
        # So we only have 4 terms total for 4x4x3.
        # 1. A00 * B00 -> C00
        # 2. A01 * B10 -> C00
        # 3. A10 * B00 -> C10
        # 4. A11 * B10 -> C10
        
        # Each term is 2x2 * 2x3.
        # Rank 11 each.
        # Total 44.
    ]
    
    # Corrected terms list:
    # (a_block, b_block, c_block)
    # a_block: 0,1,2,3 (0=A00, 1=A01, 2=A10, 3=A11)
    # b_block: 0,1 (0=B00, 1=B10)
    # c_block: 0,1 (0=C00, 1=C10)
    
    terms = [
        (0, 0, 0),
        (1, 1, 0),
        (2, 0, 1),
        (3, 1, 1)
    ]
    
    for a_blk, b_blk, c_blk in terms:
        # Strassen part (cols 0,1 of B, which are cols 0,1 of the 2x3 block)
        # B_block has cols 0,1,2.
        # Strassen uses cols 0,1.
        # Schoolbook uses col 2.
        
        # Strassen columns offset
        for c1 in range(7):
            col = col_offset + c1
            # U
            for i1 in range(2):
                for i2 in range(2):
                    i_local = i1*2 + i2
                    i_global = a_blk*2 + i_local # A index
                    val_u = U_S[i1][c1] * U_S[i2][c1] # Wait, Kronecker?
                    # No, this is a single 2x2x2 multiplication, not Kronecker.
                    # So we just use U_S directly.
                    val_u = U_S[i_local][c1]
                    if val_u != 0:
                        U[i_global][col] += val_u
            # V
            for j1 in range(2):
                for j2 in range(2):
                    j_local = j1*2 + j2
                    # B_block has cols 0,1,2.
                    # Strassen uses B cols 0,1.
                    # B index in global V:
                    # V is m x p = 4 x 7.
                    # B_block is 2x3.
                    # B00 is rows 0,1, cols 0,1,2.
                    # B10 is rows 2,3, cols 0,1,2.
                    # Strassen uses B cols 0,1.
                    # So j_global = b_blk*2 + j_local.
                    # But j_local is 0..3 for 2x2 B.
                    # B00 cols 0,1 are global cols 0,1.
                    # B10 cols 0,1 are global cols 0,1? No.
                    # B is 4x7.
                    # B00 is rows 0,1, cols 0,1,2.
                    # B10 is rows 2,3, cols 0,1,2.
                    # Strassen uses B cols 0,1.
                    # So for B00, we use global cols 0,1.
                    # For B10, we use global cols 0,1.
                    # Wait, B10 is the bottom half.
                    # So V index is (row, col).
                    # Row: b_blk*2 + j1.
                    # Col: j2 (0 or 1).
                    j_global_row = b_blk*2 + j1
                    j_global_col = j2
                    v_idx = j_global_row * p + j_global_col
                    val_v = V_S[j_local][c1]
                    if val_v != 0:
                        V[v_idx][col] += val_v
            # W
            for k1 in range(2):
                for k2 in range(2):
                    k_local = k1*2 + k2
                    k_global_row = c_blk*2 + k1
                    k_global_col = k2
                    w_idx = k_global_row * n + k_global_col
                    val_w = W_S[k_local][c1]
                    if val_w != 0:
                        W[w_idx][col] += val_w
                        
        col_offset += 7
        
        # Schoolbook part (col 2 of B)
        # 4 terms: A00*B02, A01*B12, A10*B02, A11*B12
        # A indices: a_blk*2 + 0, a_blk*2 + 1
        # B index: b_blk*2 + 0, col 2; b_blk*2 + 1, col 2
        # W indices: c_blk*2 + 0, col 2; c_blk*2 + 1, col 2
        
        # Term 1: A[0] * B[0, 2]
        col = col_offset
        u_idx = a_blk*2 + 0
        v_idx = (b_blk*2 + 0)*p + 2
        w_idx = (c_blk*2 + 0)*n + 2
        U[u_idx][col] = 1
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        col_offset += 1
        
        # Term 2: A[1] * B[1, 2]
        col = col_offset
        u_idx = a_blk*2 + 1
        v_idx = (b_blk*2 + 1)*p + 2
        w_idx = (c_blk*2 + 0)*n + 2
        U[u_idx][col] = 1
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        col_offset += 1
        
        # Term 3: A[0] * B[1, 2] -> Wait, A00 * B12?
        # A00 is rows 0,1. B12 is rows 2,3 col 2.
        # C00 is rows 0,1 col 2.
        # C00[0] = A00[0]*B00[0] + A01[0]*B10[0] ...
        # In our block decomposition:
        # C00 = A00 B00 + A01 B10.
        # A00 is 2x2. B00 is 2x3.
        # A01 is 2x2. B10 is 2x3.
        # So for A00 * B00, we use A00 rows 0,1 and B00 rows 0,1.
        # For A01 * B10, we use A01 rows 0,1 and B10 rows 2,3.
        
        # So in the term (a_blk, b_blk, c_blk):
        # A rows are a_blk*2, a_blk*2+1.
        # B rows are b_blk*2, b_blk*2+1.
        # C rows are c_blk*2, c_blk*2+1.
        
        # So:
        # Term 1: A[a_blk*2+0] * B[b_blk*2+0, 2] -> C[c_blk*2+0, 2]
        # Term 2: A[a_blk*2+1] * B[b_blk*2+1, 2] -> C[c_blk*2+0, 2]
        # Term 3: A[a_blk*2+0] * B[b_blk*2+1, 2] -> C[c_blk*2+1, 2]
        # Term 4: A[a_blk*2+1] * B[b_blk*2+0, 2] -> C[c_blk*2+1, 2]
        
        # I made a mistake above.
        # Let's redo Schoolbook for 2x2x1.
        # C[0] = A[0]B[0] + A[1]B[1]
        # C[1] = A[2]B[0] + A[3]B[1]
        # A indices: 0,1,2,3. B indices: 0,1.
        
        # Term 1: A[0] * B[0] -> C[0]
        # Term 2: A[1] * B[1] -> C[0]
        # Term 3: A[2] * B[0] -> C[1]
        # Term 4: A[3] * B[1] -> C[1]
        
        # My previous code for Term 2 was wrong.
        # Let's fix it in the code below.
        
    # Fixing the Schoolbook part logic in the loop above was tricky.
    # I will rewrite the schoolbook part clearly.
    
    # Re-initialize col_offset to 49 + 4*7 = 77
    col_offset = 49 + 28 # 77
    
    for a_blk, b_blk, c_blk in terms:
        # 4 terms
        # 1. A[0] * B[0] -> C[0]
        col = col_offset
        u_idx = a_blk*2 + 0
        v_idx = (b_blk*2 + 0)*p + 2
        w_idx = (c_blk*2 + 0)*n + 2
        U[u_idx][col] = 1
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        col_offset += 1
        
        # 2. A[1] * B[1] -> C[0]
        col = col_offset
        u_idx = a_blk*2 + 1
        v_idx = (b_blk*2 + 1)*p + 2
        w_idx = (c_blk*2 + 0)*n + 2
        U[u_idx][col] = 1
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        col_offset += 1
        
        # 3. A[2] * B[0] -> C[1]
        col = col_offset
        u_idx = a_blk*2 + 2
        v_idx = (b_blk*2 + 0)*p + 2
        w_idx = (c_blk*2 + 1)*n + 2
        U[u_idx][col] = 1
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        col_offset += 1
        
        # 4. A[3] * B[1] -> C[1]
        col = col_offset
        u_idx = a_blk*2 + 3
        v_idx = (b_blk*2 + 1)*p + 2
        w_idx = (c_blk*2 + 1)*n + 2
        U[u_idx][col] = 1
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        col_offset += 1

    return {"U": U, "V": "V", "W": W}
# EVOLVE-BLOCK-END
