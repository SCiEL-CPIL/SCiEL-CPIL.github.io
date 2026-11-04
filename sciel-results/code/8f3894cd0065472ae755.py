"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-65 exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
    
    Strategy:
    1. Decompose the 4x4 * 4x5 product into a 4x4 * 4x4 part (columns 0-3 of B) 
       and a 4x4 * 4x1 part (column 4 of B).
    2. Use recursive Strassen decomposition for the 4x4 * 4x4 part (rank 49).
    3. Use schoolbook decomposition for the 4x4 * 4x1 part (rank 16).
    Total rank = 49 + 16 = 65.
    
    The tensor convention is W[k*n + i, r] where:
    - n = 4 (rows of A / cols of output)
    - m = 4 (cols of A / rows of B)
    - p = 5 (cols of B / cols of output)
    - U has shape (n*m, rank) = (16, rank)
    - V has shape (m*p, rank) = (20, rank)
    - W has shape (p*n, rank) = (20, rank)
    """
    n, m, p = 4, 4, 5
    total_rank = 65
    
    U = [[0] * total_rank for _ in range(n * m)]  # (16, 65)
    V = [[0] * total_rank for _ in range(m * p)]  # (20, 65)
    W = [[0] * total_rank for _ in range(p * n)]  # (20, 65)
    
    # Helper: add a term to the decomposition
    def add_term(u_row, v_row, w_row, value, col):
        U[u_row][col] += value
        V[v_row][col] += value
        W[w_row][col] += value
    
    # --- Part 1: 4x4 * 4x4 (first 4 columns of B), rank 49 via Strassen ---
    # We use recursive Strassen: 4x4 = 2x2 blocks of 2x2 matrices
    # Each 2x2 * 2x2 uses Strassen with 7 multiplications
    # 7 super-multiplications * 7 sub-multiplications = 49 terms
    
    # Strassen 2x2 * 2x2 decomposition (rank 7)
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    #
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 4x4 * 4x4, we have 2x2 blocks:
    # A blocks: A[00], A[01], A[10], A[11] (each 2x2)
    # B blocks: B[00], B[01], B[10], B[11] (each 2x2)
    # C blocks: C[00], C[01], C[10], C[11] (each 2x2)
    #
    # Super-multiplications:
    # S1 = (A00 + A11) * (B00 + B11)
    # S2 = (A10 + A11) * B00
    # S3 = A00 * (B01 - B11)
    # S4 = A11 * (B10 - B00)
    # S5 = (A00 + A01) * B11
    # S6 = (A10 - A00) * (B00 + B01)
    # S7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = S1 + S4 - S5 + S7
    # C01 = S3 + S5
    # C10 = S2 + S4
    # C11 = S1 - S2 + S3 + S6
    
    # Each Sk is a 2x2 * 2x2 multiplication, computed with Strassen (7 terms)
    
    # Index mapping for 4x4 matrices:
    # A[i*m + j] where i,j in {0,1,2,3}
    # Block A[bi][bj] corresponds to A[(2*bi + di)*m + (2*bj + dj)] for di,dj in {0,1}
    
    # For the 4x4 * 4x4 part, we only use the first 4 columns of B (B[:, 0:4])
    # V index for 4x4 part: row j*p + k where j in {0,1,2,3}, k in {0,1,2,3}
    # W index for 4x4 part: k*n + i where k in {0,1,2,3}, i in {0,1,2,3}
    
    col = 0  # start column for 4x4*4x4 part
    
    # Strassen 2x2 * 2x2 coefficient table
    # Each entry: (a_coef, b_coef, c_coef) for the linear combination
    # M1: A11+B22, B11+B22, all +1
    strassen_2x2 = [
        # M1 = (A11 + A22) * (B11 + B22)
        [(1, 1, 1), (1, 1, 1), (0, 1, 1), (1, 0, 1)],  # C11 += M1, C12 += 0, C21 += 0, C22 += M1
        # M2 = (A21 + A22) * B11
        [(1, 0, 1), (1, 1, 1), (0, 0, 1), (1, 1, 1)],  # C21 += M2, C22 -= M2
        # M3 = A11 * (B12 - B22)
        [(0, 1, 1), (0, 1, 1), (1, 0, 1), (0, 1, 1)],  # C12 += M3, C22 += M3
        # M4 = A22 * (B21 - B11)
        [(1, 1, 1), (1, 0, 1), (0, 0, 1), (1, 1, 1)],  # C11 += M4, C21 += M4
        # M5 = (A11 + A12) * B22
        [(1, 0, 1), (0, 1, 1), (1, 0, 1), (0, 1, 1)],  # C11 -= M5, C12 += M5
        # M6 = (A21 - A11) * (B11 + B12)
        [(0, 0, 1), (1, 1, 1), (0, 0, 1), (1, 1, 1)],  # C22 += M6
        # M7 = (A12 - A22) * (B21 + B22)
        [(1, 0, 1), (1, 1, 1), (0, 0, 1), (1, 1, 1)],  # C11 += M7
    ]
    
    # Super-Strassen for 4x4 * 4x4
    # For each super-multiplication Sk, we need to compute it as a 2x2 * 2x2
    # Each super-multiplication involves specific blocks of A and B
    
    # Let's define the 7 super-multiplications with their block coefficients
    # Each super-multiplication Sk = (sum of A blocks with coeffs) * (sum of B blocks with coeffs)
    
    # S1 = (A00 + A11) * (B00 + B11)
    # S2 = (A10 + A11) * B00
    # S3 = A00 * (B01 - B11)
    # S4 = A11 * (B10 - B00)
    # S5 = (A00 + A01) * B11
    # S6 = (A10 - A00) * (B00 + B01)
    # S7 = (A01 - A11) * (B10 + B11)
    
    # For each Sk, the A-side is a 2x2 block matrix (linear combo of 4x4 blocks)
    # and the B-side is a 2x2 block matrix
    
    # Let me think about this more carefully.
    # A is a 4x4 matrix, viewed as a 2x2 block matrix with 2x2 blocks.
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2.
    
    # For Sk, the "left" matrix is a 2x2 block matrix where each entry is a 2x2 block
    # that is a linear combination of the original A blocks.
    # Similarly for the "right" matrix with B blocks.
    
    # Then Sk itself is a 2x2 * 2x2 multiplication (of block matrices),
    # which we compute with Strassen (7 scalar multiplications of 2x2 matrices... 
    # wait, no. Each "multiplication" in the block Strassen is a 2x2 * 2x2 matrix 
    # multiplication, which we further decompose with Strassen into 7 scalar mults.
    
    # So the structure is:
    # Level 1: 4x4 * 4x4 -> 7 block multiplications (each 2x2 * 2x2)
    # Level 2: Each 2x2 * 2x2 -> 7 scalar multiplications
    # Total: 49 scalar multiplications
    
    # Let me implement this step by step.
    
    # For the block Strassen (level 1):
    # A_blocks = [[A00, A01], [A10, A11]]
    # B_blocks = [[B00, B01], [B10, B11]]
    
    # The 7 block multiplications are:
    # M1_block = (A00 + A11) * (B00 + B11)
    # M2_block = (A10 + A11) * B00
    # M3_block = A00 * (B01 - B11)
    # M4_block = A11 * (B10 - B00)
    # M5_block = (A00 + A01) * B11
    # M6_block = (A10 - A00) * (B00 + B01)
    # M7_block = (A01 - A11) * (B10 + B11)
    
    # Each Mk_block is a 2x2 * 2x2 matrix multiplication.
    # We decompose each Mk_block using Strassen (7 terms).
    
    # For M1_block = (A00 + A11) * (B00 + B11):
    # Left matrix L = A00 + A11 (a 2x2 matrix, sum of two 2x2 blocks)
    # Right matrix R = B00 + B11 (a 2x2 matrix)
    # We compute L * R using Strassen.
    
    # L = [[L00, L01], [L10, L11]] where L00 = A00[0][0] + A11[0][0], etc.
    # R = [[R00, R01], [R10, R11]]
    
    # Strassen on L * R:
    # m1 = (L00 + L11) * (R00 + R11)
    # m2 = (L10 + L11) * R00
    # m3 = L00 * (R01 - R11)
    # m4 = L11 * (R10 - R00)
    # m5 = (L00 + L01) * R11
    # m6 = (L10 - L00) * (R00 + R01)
    # m7 = (L01 - L11) * (R10 + R11)
    
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # Now I need to express each Lij and Rij in terms of the original 4x4 matrix entries.
    
    # A00 is the top-left 2x2 block of A: A[0], A[1], A[4], A[5] (row-major)
    # A01 is the top-right 2x2 block: A[2], A[3], A[6], A[7]
    # A10 is the bottom-left 2x2 block: A[8], A[9], A[12], A[13]
    # A11 is the bottom-right 2x2 block: A[10], A[11], A[14], A[15]
    
    # Similarly for B (first 4 columns):
    # B00: B[0], B[1], B[4], B[5]
    # B01: B[2], B[3], B[6], B[7]
    # B10: B[8], B[9], B[12], B[13]
    # B11: B[10], B[11], B[14], B[15]
    
    # This is getting complex. Let me just implement it directly.
    
    # I'll create a function that adds a Strassen decomposition for a 2x2 * 2x2
    # where the 2x2 matrices are given as linear combinations of the original blocks.
    
    # Actually, let me just hard-code all 49 terms.
    
    # For each of the 7 block multiplications, I'll decompose into 7 scalar mults.
    
    # Let me define the block multiplication coefficients.
    # For M1_block = (A00 + A11) * (B00 + B11):
    # Left: L = A00 + A11
    #   L00 = A[0] + A[10]
    #   L01 = A[1] + A[11]
    #   L10 = A[4] + A[14]
    #   L11 = A[5] + A[15]
    # Right: R = B00 + B11
    #   R00 = B[0] + B[10]
    #   R01 = B[1] + B[11]
    #   R10 = B[4] + B[14]
    #   R11 = B[5] + B[15]
    
    # Strassen on L * R gives 7 terms. Each term is (a * b) where a is a linear 
    # combo of L entries and b is a linear combo of R entries.
    
    # Let me just write out all 49 terms explicitly. This is tedious but doable.
    
    # I'll use a systematic approach.
    
    # Define the 7 block multiplications as (left_blocks, right_blocks) where each
    # is a list of (block_index, coefficient) pairs.
    
    # Block indices: 0=A00, 1=A01, 2=A10, 3=A11 (same for B)
    
    block_mults = [
        # M1: (A00 + A11) * (B00 + B11)
        ([ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ]),
        # M2: (A10 + A11) * B00
        ([ (2, 1), (3, 1) ], [ (0, 1) ]),
        # M3: A00 * (B01 - B11)
        ([ (0, 1) ], [ (1, 1), (3, -1) ]),
        # M4: A11 * (B10 - B00)
        ([ (3, 1) ], [ (2, 1), (0, -1) ]),
        # M5: (A00 + A01) * B11
        ([ (0, 1), (1, 1) ], [ (3, 1) ]),
        # M6: (A10 - A00) * (B00 + B01)
        ([ (2, 1), (0, -1) ], [ (0, 1), (1, 1) ]),
        # M7: (A01 - A11) * (B10 + B11)
        ([ (1, 1), (3, -1) ], [ (2, 1), (3, 1) ]),
    ]
    
    # For each block multiplication, decompose into 7 Strassen terms.
    # Each Strassen term computes one scalar multiplication.
    
    # Strassen 2x2 * 2x2:
    # Given L = [[L00, L01], [L10, L11]], R = [[R00, R01], [R10, R11]]
    # m1 = (L00 + L11) * (R00 + R11)
    # m2 = (L10 + L11) * R00
    # m3 = L00 * (R01 - R11)
    # m4 = L11 * (R10 - R00)
    # m5 = (L00 + L01) * R11
    # m6 = (L10 - L00) * (R00 + R01)
    # m7 = (L01 - L11) * (R10 + R11)
    
    # Output:
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # For each block multiplication, I need to:
    # 1. Expand L and R entries as linear combos of original A/B entries
    # 2. For each Strassen term, expand the product as a linear combo
    # 3. Add to the appropriate C entries
    
    # This is getting very complex. Let me just build it numerically.
    
    # I'll create the U, V, W matrices by constructing each term.
    
    # For the 4x4 * 4x4 part, the output C is a 4x4 matrix.
    # C index: k*n + i where k is column (0-3), i is row (0-3)
    # So C[i][k] corresponds to W[k*4 + i]
    
    # Let me just build it step by step, adding each term.
    
    col = 0
    for blk_idx in range(7):
        left_blocks, right_blocks = block_mults[blk_idx]
        
        # Expand left matrix L (2x2) as linear combos of A entries
        # L00, L01, L10, L11 are each linear combos of A[i*m+j]
        
        # Block 0 (A00): A[0], A[1], A[4], A[5]
        # Block 1 (A01): A[2], A[3], A[6], A[7]
        # Block 2 (A10): A[8], A[9], A[12], A[13]
        # Block 3 (A11): A[10], A[11], A[14], A[15]
        
        block_to_entries = {
            0: [0, 1, 4, 5],
            1: [2, 3, 6, 7],
            2: [8, 9, 12, 13],
            3: [10, 11, 14, 15],
        }
        
        # L entries: L00, L01, L10, L11
        # Each is a sum of coefficients * A[entry]
        
        L_coeffs = [[0]*16 for _ in range(4)]  # L_coeffs[pos][a_idx]
        for block_idx, coeff in left_blocks:
            entries = block_to_entries[block_idx]
            for pos, a_idx in enumerate(entries):
                L_coeffs[pos][a_idx] += coeff
        
        # R entries: R00, R01, R10, R11
        R_coeffs = [[0]*20 for _ in range(4)]  # R_coeffs[pos][b_idx]
        for block_idx, coeff in right_blocks:
            entries = block_to_entries[block_idx]
            for pos, b_idx in enumerate(entries):
                R_coeffs[pos][b_idx] += coeff
        
        # Now compute the 7 Strassen terms
        # m1 = (L00 + L11) * (R00 + R11)
        # m2 = (L10 + L11) * R00
        # m3 = L00 * (R01 - R11)
        # m4 = L11 * (R10 - R00)
        # m5 = (L00 + L01) * R11
        # m6 = (L10 - L00) * (R00 + R01)
        # m7 = (L01 - L11) * (R10 + R11)
        
        strassen_terms = [
            # (L_pos1, L_pos2, coeff1, coeff2, R_pos1, R_pos2, coeff3, coeff4)
            (0, 1, 1, 1, 0, 1, 1, 1),  # m1
            (2, 3, 1, 1, 0, -1, 1, 0), # m2 (L10 + L11) * R00
            (0, -1, 1, 0, 1, 3, 1, -1), # m3 L00 * (R01 - R11)
            (3, -1, 1, 0, 2, 0, 1, -1), # m4 L11 * (R10 - R00)
            (0, 1, 1, 1, 3, -1, 1, 0), # m5 (L00 + L01) * R11
            (2, 0, 1, -1, 0, 1, 1, 1), # m6 (L10 - L00) * (R00 + R01)
            (1, 3, 1, -1, 2, 3, 1, 1), # m7 (L01 - L11) * (R10 + R11)
        ]
        
        # Output coefficients for C00, C01, C10, C11
        # C00 = m1 + m4 - m5 + m7
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m3 + m6
        
        output_coeffs = [
            [1, 0, 0, 1, -1, 0, 1],  # C00
            [0, 0, 1, 0, 1, 0, 0],    # C01
            [0, 1, 0, 1, 0, 0, 0],    # C10
            [1, -1, 1, 0, 0, 1, 0],   # C11
        ]
        
        for m_idx in range(7):
            L_p1, L_p2, c1, c2, R_p1, R_p2, c3, c4 = strassen_terms[m_idx]
            
            # U coefficients: linear combo of L entries
            u_coeffs = [0]*16
            if L_p1 >= 0:
                for a_idx in range(16):
                    u_coeffs[a_idx] += c1 * L_coeffs[L_p1][a_idx]
            if L_p2 >= 0:
                for a_idx in range(16):
                    u_coeffs[a_idx] += c2 * L_coeffs[L_p2][a_idx]
            
            # V coefficients: linear combo of R entries
            v_coeffs = [0]*20
            if R_p1 >= 0:
                for b_idx in range(20):
                    v_coeffs[b_idx] += c3 * R_coeffs[R_p1][b_idx]
            if R_p2 >= 0:
                for b_idx in range(20):
                    v_coeffs[b_idx] += c4 * R_coeffs[R_p2][b_idx]
            
            # W coefficients: output entries
            # C00 -> C[0][0], C[0][1], C[1][0], C[1][1] in block space
            # Which corresponds to C[0][0], C[0][2], C[2][0], C[2][2] in 4x4 space
            # Wait, no. C is a 4x4 matrix. The block structure is:
            # C = [[C00_block, C01_block], [C10_block, C11_block]]
            # C00_block is top-left 2x2: C[0][0], C[0][1], C[1][0], C[1][1]
            # C01_block is top-right 2x2: C[0][2], C[0][3], C[1][2], C[1][3]
            # C10_block is bottom-left 2x2: C[2][0], C[2][1], C[3][0], C[3][1]
            # C11_block is bottom-right 2x2: C[2][2], C[2][3], C[3][2], C[3][3]
            
            # W index: k*n + i where k is column, i is row
            # So C[i][k] -> W[k*4 + i]
            
            w_coeffs = [0]*20  # only first 16 entries used (k in 0-3)
            
            # For each output block, add the coefficient
            for out_block in range(4):
                coeff = output_coeffs[out_block][m_idx]
                if coeff == 0:
                    continue
                
                # out_block 0 = C00: rows 0-1, cols 0-1
                # out_block 1 = C01: rows 0-1, cols 2-3
                # out_block 2 = C10: rows 2-3, cols 0-1
                # out_block 3 = C11: rows 2-3, cols 2-3
                
                row_offset = (out_block // 2) * 2  # 0 for top, 2 for bottom
                col_offset = (out_block % 2) * 2   # 0 for left, 2 for right
                
                for dr in range(2):
                    for dc in range(2):
                        i = row_offset + dr
                        k = col_offset + dc
                        w_idx = k * 4 + i
                        w_coeffs[w_idx] += coeff
            
            # Add this term to U, V, W
            for a_idx in range(16):
                if u_coeffs[a_idx] != 0:
                    U[a_idx][col] += u_coeffs[a_idx]
            for b_idx in range(16):  # only first 4 cols of B
                if v_coeffs[b_idx] != 0:
                    V[b_idx][col] += v_coeffs[b_idx]
            for w_idx in range(16):
                if w_coeffs[w_idx] != 0:
                    W[w_idx][col] += w_coeffs[w_idx]
            
            col += 1
    
    # --- Part 2: 4x4 * 4x1 (5th column of B), rank 16 via schoolbook ---
    # A is 4x4, B is 4x1 (just one column: B[:, 4])
    # C is 4x1
    # C[i] = sum_j A[i][j] * B[j][4]
    
    # U: A[i*m + j] for i in 0-3, j in 0-3
    # V: B[j*p + 4] for j in 0-3, so V[j*5 + 4]
    # W: C[i] -> W[4*n + i] = W[16 + i]
    
    for i in range(4):
        for j in range(4):
            u_idx = i * 4 + j
            v_idx = j * 5 + 4
            w_idx = 4 * 4 + i  # k=4, i=i
            
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
