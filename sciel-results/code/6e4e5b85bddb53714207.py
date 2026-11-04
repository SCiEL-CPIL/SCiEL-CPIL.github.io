"""Matrix multiplication 4,4,7: rank-minimal exact tensor decomposition."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 7

    # Strassen decomposition for 2x2 x 2x2 matrix multiplication
    # Returns local (U, V, W) of shape (4,7) each with entries in {0, 1/2, 1, -1}
    def strassen_222():
        # Strassen bilinear forms:
        # M1 = (a00+a11)*(b00+b11)
        # M2 = (a10+a11)*b00
        # M3 = a00*(b01-b11)
        # M4 = a11*(b10-b00)
        # M5 = (a00+a01)*b11
        # M6 = (a10-a00)*(b00+b01)
        # M7 = (a01-a11)*(b10+b11)
        #
        # Product C = A*B:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        #
        # U matrix: rows are (a00, a01, a10, a11), columns are M1..M7
        U = [
            [1, 0, 1, 0, 1, -1, 0],   # a00
            [0, 1, 0, 0, 1, 0, -1],   # a01
            [0, 0, 1, 1, 0, 1, 0],    # a10
            [1, 0, 0, 1, 0, 0, -1],   # a11
        ]
        # V matrix: rows are (b00, b01, b10, b11), columns are M1..M7
        V = [
            [1, 1, 0, -1, 0, 1, 0],   # b00
            [0, 0, 1, 0, 0, 1, 0],    # b01
            [0, 0, 0, 1, 0, 0, 1],    # b10
            [1, 0, -1, 0, 1, 0, 1],   # b11
        ]
        # W matrix: rows are (c00, c01, c10, c11), columns are M1..M7
        W = [
            [1, 0, 0, 1, -1, 0, 1],   # c00
            [0, 1, 1, 0, 1, 0, 0],    # c01
            [0, 0, 0, 1, 0, 0, 0],    # c10
            [1, 0, 0, 0, 0, 1, 0],    # c11
        ]
        # Wait, let me re-verify the W matrix from the Strassen formulas:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        W = [
            [1, 0, 0, 1, -1, 0, 1],   # c00: M1+M4-M5+M7
            [0, 0, 1, 0, 1, 0, 0],    # c01: M3+M5
            [0, 1, 0, 1, 0, 0, 0],    # c10: M2+M4
            [1, -1, 1, 0, 0, 1, 0],   # c11: M1-M2+M3+M6
        ]
        return U, V, W

    # We will split the 4x4 matrix A into four 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # B = [[B00, B01, B02], [B10, B11, B12]] where each Bij is 2x2 (for the 4x4 part)
    #     and for the remaining 4x3 part, we use schoolbook
    
    # Actually, let's think more carefully. We have A (4x4) times B (4x7).
    # We can split B into B_left (4x4) and B_right (4x3).
    # For A * B_left, we can use a Strassen-like decomposition at the block level.
    # For A * B_right, we use schoolbook.
    
    # But actually, the problem is 4x4 * 4x7. Let's split differently.
    # Let's use the fact that we can decompose the 4x4 * 4x4 part using Strassen on 2x2 blocks,
    # and the 4x4 * 4x3 part using schoolbook on 2x2 blocks (2*2*3*2 = 24 terms per 2x2 block pair, but let's be more precise).
    
    # Better approach: 
    # Split A into 2x2 blocks: A = [[A00, A01], [A10, A11]] (each 2x2)
    # Split B (4x7) into 2x2 blocks plus a remainder:
    # B = [[B00, B01, B02], [B10, B11, B12]] where B00..B11 are 2x2, and the last column is 2x1.
    # Wait, 4x7 can't be evenly split into 2x2 blocks. 7 = 2+2+2+1 or 2+2+3 etc.
    
    # Let me try a different strategy. Let's just use the known optimal algorithms.
    # For 4x4 * 4x4, the best known rank is 49 (using Strassen recursively: 7*7=49).
    # For 4x4 * 4x3, let's figure out the rank.
    # 4x4 * 4x3 can be seen as three 4x4 * 4x1 problems.
    # A 4x4 * 4x1 multiplication: C_k = sum_j A_jk * B_k... wait no.
    # (A*B)_ik = sum_j A_ij * B_jk. For fixed k, this is a vector of 4 entries, each a dot product of a row of A and a column of B.
    # The rank of 4x4x1 (matrix-vector multiplication) is... well, it's just 16 scalar multiplications in schoolbook, but can it be done with fewer?
    # Actually, matrix-vector multiplication 4x4*4x1 is equivalent to 4 separate dot products of length 4, but they share the same A. The bilinear form has 4*4*1 = 16 terms in schoolbook. I don't think you can reduce this below 16 easily since each output depends on different combinations.
    # Wait, no. The tensor for 4x4 * 4x1 has dimension 4*4*1 = 16 in schoolbook. Can we do better? Probably not much, let's assume 16 for now.
    # So 4x4 * 4x3 would be 3 * 16 = 48 in schoolbook.
    
    # Total rank: 49 + 48 = 97.
    
    # Let's implement this.
    
    # Part 1: 4x4 * 4x4 using Strassen on 2x2 blocks (rank 49)
    # A blocks: A00, A01, A10, A11 (each 2x2)
    # B blocks: B00, B01, B10, B11 (each 2x2)
    # Strassen M terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each M_k is a product of two 2x2 matrices, which we decompose using Strassen 2x2x2x2x2 (rank 7).
    # So each M_k contributes 7 scalar multiplications.
    
    # We need to map the local Strassen U, V, W to global indices.
    
    sU, sV, sW = strassen_222()
    
    # Global U has 16 rows (A entries: a00..a33), global V has 16 rows for B_left (b00..b33), 
    # and global W has 16 rows for C_left (c00..c33).
    
    # A entry indexing: row i (0-3), col j (0-3). A_ij at index i*4+j.
    # Block A_pq (p,q in 0..1) has entries A_{2p+r, 2q+s} for r,s in 0..1.
    # So A_{2p+r, 2q+s} is at index (2p+r)*4 + (2q+s) = 8p + 4r + 2q + s.
    
    # Similarly for B_left and C_left.
    
    # For each Strassen term k (0..6), we have a linear combination of A blocks and B blocks.
    # Let's define the coefficients for A and B in each M_k.
    
    # M1: A = A00 + A11, B = B00 + B11
    # M2: A = A10 + A11, B = B00
    # M3: A = A00, B = B01 - B11
    # M4: A = A11, B = B10 - B00
    # M5: A = A00 + A01, B = B11
    # M6: A = A10 - A00, B = B00 + B01
    # M7: A = A01 - A11, B = B10 + B11
    
    # For each M_k, the A-part is a linear combination of 2x2 blocks, and the B-part similarly.
    # When we expand M_k = (sum_alpha A_block) * (sum_beta B_block) = sum_alpha sum_beta A_block * B_block,
    # each A_block * B_block is a 2x2 * 2x2 multiplication, decomposed with Strassen rank 7.
    # The coefficient for each scalar multiplication is alpha * beta * (local Strassen coefficient).
    
    # Let's build the global U, V, W for the 4x4*4x4 part.
    
    # Initialize with zeros
    U_44 = [[0.0] * 49 for _ in range(16)]
    V_44 = [[0.0] * 49 for _ in range(16)]
    W_44 = [[0.0] * 49 for _ in range(16)]
    
    # For each Strassen term k (0..6):
    for k in range(7):
        # Determine A block coefficients
        a_coeff = {}  # (p,q) -> coeff
        if k == 0:  # M1
            a_coeff = {(0,0): 1, (1,1): 1}
        elif k == 1:  # M2
            a_coeff = {(1,0): 1, (1,1): 1}
        elif k == 2:  # M3
            a_coeff = {(0,0): 1}
        elif k == 3:  # M4
            a_coeff = {(1,1): 1}
        elif k == 4:  # M5
            a_coeff = {(0,0): 1, (0,1): 1}
        elif k == 5:  # M6
            a_coeff = {(1,0): 1, (0,0): -1}
        elif k == 6:  # M7
            a_coeff = {(0,1): 1, (1,1): -1}
        
        # Determine B block coefficients
        b_coeff = {}  # (p,q) -> coeff
        if k == 0:  # M1
            b_coeff = {(0,0): 1, (1,1): 1}
        elif k == 1:  # M2
            b_coeff = {(0,0): 1}
        elif k == 2:  # M3
            b_coeff = {(0,1): 1, (1,1): -1}
        elif k == 3:  # M4
            b_coeff = {(1,0): 1, (0,0): -1}
        elif k == 4:  # M5
            b_coeff = {(1,1): 1}
        elif k == 5:  # M6
            b_coeff = {(0,0): 1, (0,1): 1}
        elif k == 6:  # M7
            b_coeff = {(1,0): 1, (1,1): 1}
        
        # For each pair of A block (pa, qa) and B block (pb, qb)
        for (pa, qa), ca in a_coeff.items():
            for (pb, qb), cb in b_coeff.items():
                # A block A_pa,qa has entries A_{2pa+r, 2qa+s} for r,s in 0..1
                # B block B_pb,qb has entries B_{2pb+r, 2qb+s} for r,s in 0..1
                # The product A_pa,qa * B_pb,qb is a 2x2 * 2x2 multiplication.
                # We decompose this using Strassen 2x2.
                # Local Strassen U, V, W have 4 rows each (a00,a01,a10,a11) etc.
                # Local row index for A entry (r,s): r*2+s
                # Local row index for B entry (r,s): r*2+s
                # Local row index for C entry (r,s): r*2+s
                
                for local_col in range(7):
                    global_col = k * 7 + local_col
                    
                    # Add contributions from A
                    for r in range(2):
                        for s in range(2):
                            local_a_row = r * 2 + s
                            global_a_row = (2*pa + r) * 4 + (2*qa + s)
                            U_44[global_a_row][global_col] += ca * sU[local_a_row][local_col]
                    
                    # Add contributions from B
                    for r in range(2):
                        for s in range(2):
                            local_b_row = r * 2 + s
                            global_b_row = (2*pb + r) * 4 + (2*qb + s)
                            V_44[global_b_row][global_col] += cb * sV[local_b_row][local_col]
                    
                    # Add contributions to C (for the 4x4*4x4 part, C = A*B_left)
                    # C block C_pc,qc where pc=pa, qc=qb (since A_pa,qa * B_pb,qb contributes to C_pa,qa? No!)
                    # Wait, I made an error. The Strassen formulas give C blocks in terms of A and B blocks.
                    # C = A*B, so C_pq = sum_r A_pr * B_rq.
                    # In Strassen, the M_k terms are used to reconstruct C00, C01, C10, C11.
                    # So the output blocks are C00, C01, C10, C11, not C_pa,qa for arbitrary pa,qa.
                    
                    # I need to redo this. The Strassen decomposition computes C = A*B where both are 4x4,
                    # and the result is expressed in terms of the 4 output blocks C00, C01, C10, C11.
                    # So the W matrix should map to C00, C01, C10, C11, not to C_pa,qa.
                    
                    pass  # We'll handle W differently
    
    # Let me redo the W part. The Strassen output C = A*B has blocks:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So for each output block C_pq (p,q in 0..1), the contribution from M_k is:
    # C00: M1:+1, M4:+1, M5:-1, M7:+1
    # C01: M3:+1, M5:+1
    # C10: M2:+1, M4:+1
    # C11: M1:+1, M2:-1, M3:+1, M6:+1
    
    # So for global W, for each output block (pc, qc), and each M_k, we add the Strassen local W contribution
    # scaled by the coefficient of M_k in C_pc,qc.
    
    # Reset W_44
    W_44 = [[0.0] * 49 for _ in range(16)]
    
    c_coeff = {}  # (pc,qc) -> {k: coeff}
    c_coeff[(0,0)] = {0: 1, 3: 1, 4: -1, 6: 1}
    c_coeff[(0,1)] = {2: 1, 4: 1}
    c_coeff[(1,0)] = {1: 1, 3: 1}
    c_coeff[(1,1)] = {0: 1, 1: -1, 2: 1, 5: 1}
    
    for (pc, qc), k_coeffs in c_coeff.items():
        for k, coeff in k_coeffs.items():
            # C block C_pc,qc has entries C_{2pc+r, 2qc+s} for r,s in 0..1
            # Local C entry (r,s) is at local row r*2+s
            for r in range(2):
                for s in range(2):
                    local_c_row = r * 2 + s
                    global_c_row = (2*pc + r) * 4 + (2*qc + s)
                    for local_col in range(7):
                        global_col = k * 7 + local_col
                        W_44[global_c_row][global_col] += coeff * sW[local_c_row][local_col]
    
    # Part 2: 4x4 * 4x3 using schoolbook on 2x2 blocks
    # A is 4x4, split into 2x2 blocks: A00, A01, A10, A11
    # B_right is 4x3. We can split this into a 4x2 part and a 4x1 part, or handle it directly.
    # Let's just do schoolbook at the scalar level for the 4x4 * 4x3 part.
    # That's 4*4*3 = 48 scalar multiplications.
    
    # Global rank = 49 + 48 = 97
    total_rank = 97
    
    # Initialize full U, V, W
    U = [[0.0] * total_rank for _ in range(16)]  # A has 16 entries
    V = [[0.0] * total_rank for _ in range(28)]  # B has 4*7=28 entries
    W = [[0.0] * total_rank for _ in range(28)]  # C has 4*7=28 entries
    
    # Copy in the 4x4*4x4 part (first 49 columns)
    for i in range(16):
        for j in range(49):
            U[i][j] = U_44[i][j]
            V[i][j] = V_44[i][j]
            W[i][j] = W_44[i][j]
    
    # Add the 4x4*4x3 schoolbook part (next 48 columns)
    # (A*B)_ik = sum_j A_ij * B_jk for i in 0..3, k in 4..6
    # Wait, B is 4x7, so B_jk for j in 0..3, k in 0..6.
    # C is 4x7, so C_ik for i in 0..3, k in 0..6.
    # The 4x4*4x4 part gives C_ik for k in 0..3.
    # The 4x4*4x3 part gives C_ik for k in 4..6.
    
    col_offset = 49
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                U[i * 4 + j][col_offset] = 1.0
                V[j * 7 + k][col_offset] = 1.0
                W[k * 4 + i][col_offset] = 1.0
                col_offset += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
