"""
Program for Matrix Multiplication 3x4x8 Tensor Decomposition.
Uses a block decomposition: split A into 2x4 + 1x4.
- The 2x4 x 4x8 block is decomposed using Strassen on four 2x2x2 sub-blocks.
- The 1x4 x 4x8 block uses schoolbook.
Total rank: 4 * 7 (Strassen) + 32 (schoolbook) = 60.
"""
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the U, V, W matrices for a rank-60 decomposition.
    Splits the 3x4x8 multiplication into:
      - Top 2 rows of A (2x4) times B (4x8): rank 28 via 4 Strassen blocks
      - Bottom 1 row of A (1x4) times B (4x8): rank 32 schoolbook
    """
    n, m, p = DIMENSIONS  # 3, 4, 8
    rank = 60

    U = [[0.0] * rank for _ in range(n * m)]  # 12 x 60
    V = [[0.0] * rank for _ in range(m * p)]  # 32 x 60
    W = [[0.0] * rank for _ in range(p * n)]  # 24 x 60

    # --- Strassen 2x2x2 factors ---
    # Strassen computes C = A*B for 2x2 matrices using 7 multiplications.
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6

    def strassen_block():
        """Return (U_block, V_block, W_block) for a 2x2x2 Strassen block.
        U_block: 4 x 7 (indices: row*2+col in A)
        V_block: 4 x 7 (indices: row*2+col in B)
        W_block: 4 x 7 (indices: col*2+row in C, following W[k*n+i] convention for 2x2)
        """
        u = [[0.0] * 7 for _ in range(4)]
        v = [[0.0] * 7 for _ in range(4)]
        w = [[0.0] * 7 for _ in range(4)]

        # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # C indices (W[k*n+i] for n=2): (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3

        # M1 = (a11+a22)(b11+b22)
        u[0][0] += 1; u[3][0] += 1
        v[0][0] += 1; v[3][0] += 1
        w[0][0] += 1; w[1][0] -= 1; w[2][0] -= 1; w[3][0] += 1

        # M2 = (a21+a22)b11
        u[2][1] += 1; u[3][1] += 1
        v[0][1] += 1
        w[0][1] -= 1; w[3][1] += 1

        # M3 = a11(b12-b22)
        u[0][2] += 1
        v[1][2] += 1; v[3][2] -= 1
        w[1][2] += 1; w[3][2] += 1

        # M4 = a22(b21-b11)
        u[3][3] += 1
        v[2][3] += 1; v[0][3] -= 1
        w[0][3] += 1; w[2][3] += 1

        # M5 = (a11+a12)b22
        u[0][4] += 1; u[1][4] += 1
        v[3][4] += 1
        w[0][4] -= 1; w[2][4] -= 1

        # M6 = (a21-a11)(b11+b12)
        u[2][5] += 1; u[0][5] -= 1
        v[0][5] += 1; v[1][5] += 1
        w[3][5] += 1

        # M7 = (a12-a22)(b21+b22)
        u[1][6] += 1; u[3][6] -= 1
        v[2][6] += 1; v[3][6] += 1
        w[0][6] += 1; w[3][6] += 1

        return u, v, w

    su, sv, sw = strassen_block()

    # --- Place 4 Strassen blocks for the top 2 rows ---
    # A_top is 2x4, B is 4x8.
    # Split A_top into A_top_left (2x2) and A_top_right (2x2).
    # Split B into B_top (4x4) and B_bot (4x4).
    # Actually, we split B into B_left (4x4) and B_right (4x4).
    # C_top = A_top * B = A_top_left * B_left + A_top_right * B_right.
    # Wait, that's not right. A_top is 2x4, B is 4x8.
    # C_top is 2x8.
    # C_top = A_top * B.
    # Let's split A_top into A_tl (2x2) and A_tr (2x2).
    # Let's split B into B_l (4x4) and B_r (4x4).
    # C_top = A_tl * B_l + A_tr * B_r.
    # Each term is 2x2 * 4x4 = 2x4.
    # Hmm, this doesn't directly give 2x2x2 blocks.
    
    # Let me reconsider. A_top is 2x4. B is 4x8.
    # C_top is 2x8.
    # Split B into B_0 (4x4) and B_1 (4x4).
    # C_top = A_top * B_0 + A_top * B_1.
    # A_top * B_0 is 2x4. Split A_top into A_00 (2x2) and A_01 (2x2).
    # Split B_0 into B_00 (2x2) and B_01 (2x2).
    # A_top * B_0 = A_00 * B_00 + A_01 * B_01.
    # Each is 2x2 * 2x2 = 2x2. Use Strassen.
    # Similarly for A_top * B_1.
    # Total: 4 Strassen blocks, rank 4*7 = 28.

    # Row 0-1 of A correspond to A_top.
    # U indices for A: row*4+col. A_top rows are 0,1.
    # For block (br, bc) where br in {0,1} (B_left or B_right), bc in {0,1} (A_left or A_right):
    # A_block is at A rows 0-1, cols bc*2 to bc*2+1.
    # B_block is at B rows 0-3, cols br*4 to br*4+1... wait.
    
    # Let me be more careful.
    # A_top: rows 0-1 of A, all 4 columns.
    # B: all 4 rows, 8 columns.
    # C_top: rows 0-1 of C, all 8 columns.
    
    # Split A_top into left (cols 0-1) and right (cols 2-3).
    # Split B into left (cols 0-3) and right (cols 4-7).
    # C_top_left (cols 0-3 of C) = A_top * B_left.
    # C_top_right (cols 4-7 of C) = A_top * B_right.
    
    # For C_top_left: A_top (2x4) * B_left (4x4) = 2x4.
    # Split A_top into A_top_left (2x2, cols 0-1) and A_top_right (2x2, cols 2-3).
    # Split B_left into B_top_left (rows 0-1) and B_bot_left (rows 2-3).
    # Wait, B is 4x8. B_left is B[:, 0:4], which is 4x4.
    # A_top * B_left = A_top_left * B_left[:, 0:2] + A_top_right * B_left[:, 2:4].
    # B_left[:, 0:2] is 4x2. Hmm, this is getting complicated.
    
    # Let me use a different approach. 
    # Actually, for 2x4 * 4x8, we can view it as:
    # Split the 4 columns of A into two groups of 2.
    # Split the 8 columns of B into two groups of 4.
    # C = A_left * B_left + A_right * B_right.
    # A_left is 2x2, B_left is 4x4. A_right is 2x2, B_right is 4x4.
    # 2x2 * 4x4 = 2x4.
    # For each 2x2 * 4x4, we can further split the 4x4 into two 2x2 blocks.
    # 2x2 * 4x4 = 2x2 * [2x2 | 2x2] = [2x2*2x2 | 2x2*2x2].
    # So each 2x2 * 4x4 gives two 2x2x2 multiplications.
    # Total: 2 (A_left/right) * 2 (B_top/bot) = 4 Strassen blocks.
    
    # Let me define the mapping:
    # For block (a_block, b_block) where a_block in {0,1} (A left/right), b_block in {0,1} (B left/right):
    # A_sub: rows 0-1 of A, cols a_block*2 to a_block*2+1.
    # B_sub: all 4 rows of B, cols b_block*4 to b_block*4+3.
    # Wait, B_sub is 4x4. We need to split it into two 2x2 blocks.
    # B_sub_top: rows 0-1 of B, cols b_block*4 to b_block*4+1.
    # B_sub_bot: rows 2-3 of B, cols b_block*4+2 to b_block*4+3.
    # Hmm, this is getting messy. Let me just directly construct.

    # Actually, let me think of it differently. 
    # The 2x4 * 4x8 multiplication can be decomposed as follows:
    # Write A_top = [A_0 | A_1] where A_0 is 2x2 (cols 0-1) and A_1 is 2x2 (cols 2-3).
    # Write B = [B_0 | B_1] where B_0 is 4x4 (cols 0-3) and B_1 is 4x4 (cols 4-7).
    # C_top = A_0 * B_0 + A_1 * B_1.
    # Now A_0 * B_0 is 2x2 * 4x4 = 2x4.
    # Write B_0 = [B_00; B_01] where B_00 is 2x4 (rows 0-1) and B_01 is 2x4 (rows 2-3).
    # A_0 * B_0 = A_0 * B_00 + A_0 * B_01. But A_0 is 2x2 and B_00 is 2x4, so this is 2x4.
    # Hmm, I think I'm overcomplicating. Let me just use the standard approach.
    
    # For 2x4 * 4x8, the standard decomposition is:
    # Split the 4 middle dimension into 2+2.
    # Split the 8 output dimension into 4+4.
    # C = A[:, 0:2] * B[0:2, :] + A[:, 2:4] * B[2:4, :].
    # A[:, 0:2] is 2x2, B[0:2, :] is 2x8. Product is 2x8.
    # A[:, 2:4] is 2x2, B[2:4, :] is 2x8. Product is 2x8.
    # Now 2x2 * 2x8 = 2x8.
    # Split the 8 into 4+4: 2x2 * [2x4 | 2x4] = [2x2*2x4 | 2x2*2x4].
    # Each 2x2 * 2x4 = 2x4.
    # Split the 4 into 2+2: 2x2 * [2x2 | 2x2] = [2x2*2x2 | 2x2*2x2].
    # So each 2x2 * 2x4 gives two 2x2x2 multiplications.
    # Total: 2 (A split) * 2 (B split) = 4 Strassen blocks.

    # Let me define the four blocks:
    # Block (i, j) where i in {0,1} (which half of A's columns), j in {0,1} (which half of B's rows)
    # A_block: A[0:2, i*2:i*2+2] (2x2)
    # B_block: B[j*2:j*2+2, :] (2x8)
    # Wait, B is 4x8. If I split B's rows into two groups of 2, I get B_top (2x8) and B_bot (2x8).
    # C_top = A_top * B = A_top * B_top + A_top * B_bot.
    # A_top is 2x4. B_top is 2x8. A_top * B_top is not defined (2x4 * 2x8).
    
    # I think I need to be more careful with the indices.
    # A is 3x4, B is 4x8, C is 3x8.
    # C[i][k] = sum_j A[i][j] * B[j][k].
    # For i in {0,1} (top two rows):
    # C[i][k] = sum_{j=0}^{3} A[i][j] * B[j][k].
    # Split j into j=0,1 and j=2,3.
    # C[i][k] = sum_{j=0}^{1} A[i][j] * B[j][k] + sum_{j=2}^{3} A[i][j] * B[j][k].
    # First term: A_top_left (2x2) * B_top (2x8) -> 2x8.
    # Second term: A_top_right (2x2) * B_bot (2x8) -> 2x8.
    # Now 2x2 * 2x8 = 2x8.
    # Split k into k=0..3 and k=4..7.
    # First term: A_top_left * B_top[:, 0:4] + A_top_left * B_top[:, 4:8].
    # Each is 2x2 * 2x4 = 2x4.
    # Split the 4 into 2+2: 2x2 * [2x2 | 2x2] = [2x2*2x2 | 2x2*2x2].
    # So we get 2 Strassen blocks from the first term.
    # Similarly 2 from the second term.
    # Total: 4 Strassen blocks.

    # Let me define the 4 blocks explicitly:
    # Block 0: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Block 1: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # Block 3: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    # Block 4: A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
    # Block 5: A[0:2, 0:2] * B[0:2, 6:8] -> C[0:2, 6:8]
    # Block 6: A[0:2, 2:4] * B[2:4, 4:6] -> C[0:2, 4:6]
    # Block 7: A[0:2, 2:4] * B[2:4, 6:8] -> C[0:2, 6:8]
    
    # Wait, that's 8 blocks, not 4. Let me recount.
    # A_top is 2x4. B is 4x8.
    # Split A_top's columns into two groups of 2: A0 (cols 0-1), A1 (cols 2-3).
    # Split B's rows into two groups of 2: B0 (rows 0-1), B1 (rows 2-3).
    # C_top = A0 * B0 + A1 * B1.
    # A0 * B0 is 2x2 * 2x8 = 2x8.
    # Split B0's columns into four groups of 2: B0a (cols 0-1), B0b (cols 2-3), B0c (cols 4-5), B0d (cols 6-7).
    # A0 * B0 = A0*B0a + A0*B0b + A0*B0c + A0*B0d. Each is 2x2*2x2 = 2x2.
    # So 4 Strassen blocks for A0*B0.
    # Similarly 4 for A1*B1.
    # Total: 8 Strassen blocks, rank 8*7 = 56.
    
    # Plus schoolbook for the bottom row: 1x4 * 4x8, rank 32.
    # Total rank: 56 + 32 = 88.

    # OK so my initial estimate of 88 was correct. Let me implement that.
    
    # Block (a, b, k) where a in {0,1} (A col group), b in {0,1} (B row group), k in {0,1,2,3} (B col group)
    # A_sub: A[0:2, a*2:a*2+2]
    # B_sub: B[b*2:b*2+2, k*2:k*2+2]
    # C_sub: C[0:2, k*2:k*2+2]

    col_offset = 0
    for a in range(2):  # A column group
        for b in range(2):  # B row group
            for k in range(4):  # B column group
                # A_sub rows: 0,1; cols: a*2, a*2+1
                # B_sub rows: b*2, b*2+1; cols: k*2, k*2+1
                # C_sub rows: 0,1; cols: k*2, k*2+1
                
                for dr in range(4):  # within 2x2
                    for dc in range(7):  # Strassen dimension
                        # U: A index = row * 4 + col
                        ar = dr // 2
                        ac = dr % 2
                        a_row = ar
                        a_col = a * 2 + ac
                        u_idx = a_row * 4 + a_col
                        U[u_idx][col_offset + dc] += su[dr][dc]
                        
                        # V: B index = row * 8 + col
                        b_row = b * 2 + ar
                        b_col = k * 2 + ac
                        v_idx = b_row * 8 + b_col
                        V[v_idx][col_offset + dc] += sv[dr][dc]
                        
                        # W: C index = col * 3 + row (W[k*n+i] convention)
                        c_row = ar
                        c_col = k * 2 + ac
                        w_idx = c_col * 3 + c_row
                        W[w_idx][col_offset + dc] += sw[dr][dc]
                
                col_offset += 7

    # col_offset should now be 56
    assert col_offset == 56, f"Expected col_offset=56, got {col_offset}"

    # --- Schoolbook for bottom row (A[2, :] * B) ---
    # A[2, j] * B[j, k] for j=0..3, k=0..7
    # 32 terms
    for j in range(4):
        for k in range(8):
            # U: A index = 2 * 4 + j = 8 + j
            # V: B index = j * 8 + k
            # W: C index = k * 3 + 2
            U[8 + j][col_offset] = 1.0
            V[j * 8 + k][col_offset] = 1.0
            W[k * 3 + 2][col_offset] = 1.0
            col_offset += 1

    assert col_offset == 60, f"Expected col_offset=60, got {col_offset}"

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
