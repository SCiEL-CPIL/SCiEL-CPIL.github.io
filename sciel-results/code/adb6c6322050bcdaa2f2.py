"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-42 exact tensor decomposition for 2x4 * 4x6 matrix multiplication.
    
    The decomposition is built from:
    - 2 Strassen blocks (each rank 7) for the 2x2 sub-problems, giving rank 14
    - 28 additional rank-1 terms for the remaining (non-Strassen) parts
    
    The key insight: we can partition the 2x4x6 problem into blocks and use
    Strassen's algorithm for the 2x2 block multiplications, which reduces the
    total rank from the schoolbook 48 to 42.
    
    Structure:
    - A is 2x4, B is 4x6, C is 2x6
    - Split A into columns: A = [A1, A2] where A1 is 2x2, A2 is 2x2
    - Split B into rows: B = [B1; B2] where B1 is 2x6, B2 is 2x6
    - C = A1*B1 + A2*B2
    
    For each 2x2 * 2x6 multiplication, we can use Strassen-style tricks.
    A 2x2 * 2x2 needs rank 7 with Strassen. A 2x2 * 2x6 = (2x2*2x2) + (2x2*2x2)
    so naively 14, but we can share some structure.
    
    Actually, let me think more carefully. The standard approach:
    - C = A*B where A is 2x4, B is 4x6
    - Write A = [A_left, A_right] (each 2x2), B = [B_top; B_bottom] (each 2x6)
    - C = A_left * B_top + A_right * B_bottom
    
    Each term A_left * B_top is (2x2)*(2x6). We can write B_top = [B_top1, B_top2]
    where B_top1 is 2x3 and B_top2 is 2x3. Then A_left * B_top = [A_left*B_top1, A_left*B_top2].
    
    A_left*B_top1 is (2x2)*(2x3) = 2x3 result. This has 6 entries, schoolbook rank 12.
    With Strassen for the 2x2 part: we can do (2x2)*(2x1) with rank 7? No, that's not right.
    
    Let me reconsider. The tensor rank of (n x m) * (m x p) multiplication:
    - For 2x2 * 2x2: rank 7 (Strassen)
    - For 2x2 * 2x1: rank 5 (known result)
    - For 2x2 * 1x2: rank 5 (by duality)
    - For 2x1 * 1x2: rank 1 (outer product)
    
    Actually, let me use a different decomposition strategy.
    
    A (2x4) * B (4x6) = C (2x6)
    
    Split into 4 blocks of (2x2)*(2x3) or similar... 
    
    Let me use the approach from the measured knowledge: rank-42 via Strassen blocks.
    The decomposition uses 6 Strassen blocks (each rank 7) = rank 42.
    
    How to get 6 Strassen blocks from a 2x4 * 4x6 problem?
    
    A is 2x4. Split into 2 column blocks of 2: A = [A1 | A2], each 2x2.
    B is 4x6. Split into 3 row blocks... wait, B is 4x6.
    
    Let me split B into 2 row blocks of 2: B = [B1; B2], each 2x6.
    Then C = A1*B1 + A2*B2.
    
    Now B1 is 2x6. Split B1 into 3 column blocks of 2: B1 = [B1a | B1b | B1c], each 2x2.
    Similarly B2 = [B2a | B2b | B2c].
    
    C = A1*[B1a|B1b|B1c] + A2*[B2a|B2b|B2c]
      = [A1*B1a + A2*B2a | A1*B1b + A2*B2b | A1*B1c + A2*B2c]
    
    Each block Ai*Bji is (2x2)*(2x2), which can be done with Strassen (rank 7).
    There are 6 such blocks: A1*B1a, A1*B1b, A1*B1c, A2*B2a, A2*B2b, A2*B2c.
    Total rank = 6 * 7 = 42.
    
    This is valid! Each (2x2)*(2x2) sub-problem is independent and uses Strassen.
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # We'll build the decomposition by composing Strassen blocks.
    # 
    # For each pair (i in {0,1}, j in {0,1,2}), we have a (2x2)*(2x2) multiplication
    # A[i] * B[j] contributing to C[:, 2*j:2*j+2]
    # 
    # A[i] is rows 0..1, cols 2*i..2*i+1 of A
    # B[j] is rows 2*j..2*j+1, cols 0..1 of B... wait, no.
    # 
    # Let me re-index properly.
    # A is n x m = 2 x 4. A[i,j] for i in 0..1, j in 0..3.
    # B is m x p = 4 x 6. B[i,j] for i in 0..3, j in 0..5.
    # C is n x p = 2 x 6. C[i,j] for i in 0..1, j in 0..5.
    #
    # C[i,j] = sum_{k=0}^{3} A[i,k] * B[k,j]
    #
    # Split k into two groups: k in {0,1} and k in {2,3}
    # C[i,j] = sum_{k=0}^{1} A[i,k]*B[k,j] + sum_{k=2}^{3} A[i,k]*B[k,j]
    #
    # Let A0[i,:] = A[i, 0:2] (2x2), A1[i,:] = A[i, 2:4] (2x2)
    # Let B0[:,j] = B[0:2, j] (2x6), B1[:,j] = B[2:4, j] (2x6)
    #
    # C = A0 * B0 + A1 * B1
    #
    # Now split each 2x6 into three 2x2 blocks:
    # B0 = [B0_0 | B0_1 | B0_2] where B0_t is 2x2 (cols 2t, 2t+1)
    # B1 = [B1_0 | B1_1 | B1_2]
    #
    # C = [A0*B0_0 + A1*B1_0 | A0*B0_1 + A1*B1_1 | A0*B0_2 + A1*B1_2]
    #
    # Each A_t * B_u_t is (2x2)*(2x2) -> (2x2), done with Strassen rank 7.
    # Total: 6 * 7 = 42.
    
    # Strassen decomposition for 2x2 * 2x2:
    # Given X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # Z = X*Y = [[z00, z01], [z10, z11]]
    #
    # m1 = x00 * (y11 - y10)
    # m2 = (x00 + x01) * y10
    # m3 = (x10 + x11) * y00
    # m4 = x11 * (y01 - y00)
    # m5 = (x00 + x10) * (y00 + y11)
    # m6 = (x01 - x10) * (y10 + y11)
    # m7 = (x11 - x00) * (y00 + y01)
    #
    # z00 = m1 + m4 + m5 - m6 - m7
    # z01 = m2 + m7
    # z10 = m3 + m6
    # z11 = m1 + m3 - m2 + m4
    
    # For our tensor decomposition, we need U, V, W matrices.
    # U has shape (n*m, rank) = (8, 42)
    # V has shape (m*p, rank) = (24, 42)  
    # W has shape (p*n, rank) = (12, 42)
    #
    # Indexing:
    # U[i*m + j, c] corresponds to A[i,j]
    # V[j*p + k, c] corresponds to B[j,k]
    # W[k*n + i, c] corresponds to C[k,i]... wait, the convention is W[k*n+i, r]
    # 
    # The trace convention: W[k*n+i, r] means the (k*n+i)-th row of W.
    # This corresponds to C[i, k] where i is row of C, k is col of C.
    # So W index = k*n + i where k is column of C (0..5), i is row of C (0..1).
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    # Strassen formulas for 2x2*2x2:
    # For matrices X (rows x, cols y) and Y (rows y, cols z):
    # X indices: x00=0, x01=1, x10=2, x11=3 (row-major: row0=[x00,x01], row1=[x10,x11])
    # Y indices: y00=0, y01=1, y10=2, y11=3
    # Z indices: z00=0, z01=1, z10=2, z11=3
    #
    # m1 = x00 * (y11 - y10): U coeff for x00=1, V coeff for y11=1, y10=-1, W coeff for z terms
    # m2 = (x00+x01) * y10: U coeff x00=1,x01=1, V coeff y10=1
    # m3 = (x10+x11) * y00: U coeff x10=1,x11=1, V coeff y00=1
    # m4 = x11 * (y01-y00): U coeff x11=1, V coeff y01=1,y00=-1
    # m5 = (x00+x10)*(y00+y11): U coeff x00=1,x10=1, V coeff y00=1,y11=1
    # m6 = (x01-x10)*(y10+y11): U coeff x01=1,x10=-1, V coeff y10=1,y11=1
    # m7 = (x11-x00)*(y00+y01): U coeff x11=1,x00=-1, V coeff y00=1,y01=1
    #
    # Reconstruction:
    # z00 = m1 + m4 + m5 - m6 - m7
    # z01 = m2 + m7
    # z10 = m3 + m6
    # z11 = m1 + m3 - m2 + m4
    
    # Strassen multipliers: (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: tuple of (x_index, coeff) for X
    # V_coeffs: tuple of (y_index, coeff) for Y
    # W_coeffs: tuple of (z_index, coeff) for Z
    strassen = [
        # m1: x00 * (y11 - y10)
        ((0, 1), (3, 1), (2, -1), [(0, 1), (3, 1)]),   # z00+=1, z11+=1
        # m2: (x00+x01) * y10
        ((0, 1), (1, 1), (2, 1), [(1, 1), (2, 0)]),   # z01+=1
        # Wait, let me redo this more carefully.
        
        # m1: x00 * (y11 - y10)
        # z00 += m1, z11 += m1
        ((0, 1), (3, 1), (2, -1), {0: 1, 3: 1}),
        # m2: (x00+x01) * y10
        # z01 += m2
        ((0, 1), (1, 1), (2, 1), {1: 1}),
        # m3: (x10+x11) * y00
        # z10 += m3, z11 += m3
        ((2, 1), (3, 1), (0, 1), {2: 1, 3: 1}),
        # m4: x11 * (y01-y00)
        # z00 += m4, z11 += m4
        ((3, 1), (1, 1), (0, -1), {0: 1, 3: 1}),
        # m5: (x00+x10)*(y00+y11)
        # z00 += m5
        ((0, 1), (2, 1), (0, 1), (3, 1), {0: 1}),
        # m6: (x01-x10)*(y10+y11)
        # z00 -= m6, z10 += m6
        ((1, 1), (2, -1), (2, 1), (3, 1), {0: -1, 2: 1}),
        # m7: (x11-x00)*(y00+y01)
        # z00 -= m7, z01 += m7
        ((3, 1), (0, -1), (0, 1), (1, 1), {0: -1, 1: 1}),
    ]
    
    # Let me define strassen more cleanly as a list of 7 tuples:
    # Each tuple: (u_dict, v_dict, w_dict)
    # u_dict: maps local X index (0-3) to coefficient
    # v_dict: maps local Y index (0-3) to coefficient  
    # w_dict: maps local Z index (0-3) to coefficient
    
    strassen_defs = [
        # m1
        ({0: 1}, {3: 1, 2: -1}, {0: 1, 3: 1}),
        # m2
        ({0: 1, 1: 1}, {2: 1}, {1: 1}),
        # m3
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: 1}),
        # m4
        ({3: 1}, {1: 1, 0: -1}, {0: 1, 3: 1}),
        # m5
        ({0: 1, 2: 1}, {0: 1, 3: 1}, {0: 1}),
        # m6
        ({1: 1, 2: -1}, {2: 1, 3: 1}, {0: -1, 2: 1}),
        # m7
        ({3: 1, 0: -1}, {0: 1, 1: 1}, {0: -1, 1: 1}),
    ]
    
    col = 0
    # For each (a_block, b_block) pair:
    # a_block in {0, 1}: which 2x2 block of A (columns 2*a_block to 2*a_block+1)
    # b_block in {0, 1, 2}: which 2x2 block of B (columns 2*b_block to 2*b_block+1)
    # The rows of B used are 2*a_block to 2*a_block+1 (since C = A0*B0 + A1*B1)
    # 
    # Wait, I need to be more careful.
    # C = A0 * B0 + A1 * B1
    # A0 is A[:, 0:2], B0 is B[0:2, :]
    # A1 is A[:, 2:4], B1 is B[2:4, :]
    #
    # For the term A0 * B0:
    #   A0[i, j] = A[i, j] for i in 0..1, j in 0..1
    #   B0[k, l] = B[k, l] for k in 0..1, l in 0..5
    #   Contribution to C[i, l] = sum_{k=0}^{1} A[i,k] * B[k,l]
    #
    # Split B0 into 3 blocks of 2 columns: B0_t = B0[:, 2*t:2*t+2] for t=0,1,2
    #   B0_t[k, s] = B[k, 2*t + s] for k in 0..1, s in 0..1
    #
    # Then A0 * B0_t is (2x2)*(2x2) = (2x2)
    #   Result R[i, s] = sum_{k=0}^{1} A0[i,k] * B0_t[k,s] = sum_{k=0}^{1} A[i,k] * B[k, 2t+s]
    #   This contributes to C[i, 2t+s]
    #
    # Similarly for A1 * B1_t where B1_t = B1[:, 2*t:2*t+2] = B[2:4, 2t:2t+2]
    #   B1_t[k, s] = B[2+k, 2t+s] for k in 0..1, s in 0..1
    #   Result R[i, s] = sum_{k=0}^{1} A1[i,k] * B1_t[k,s] = sum_{k=0}^{1} A[i,2+k] * B[2+k, 2t+s]
    #   This contributes to C[i, 2t+s]
    #
    # So for each (a_block, t) pair, we have a (2x2)*(2x2) Strassen block.
    # a_block in {0, 1}, t in {0, 1, 2}: total 6 blocks, each rank 7, total rank 42.
    
    for a_block in range(2):
        for t in range(3):
            for s_idx in range(7):
                u_dict, v_dict, w_dict = strassen_defs[s_idx]
                
                # U: A indices
                # A local index (i, j) -> global A index i*m + j = i*4 + j
                # For a_block=0: j_global = j (0 or 1)
                # For a_block=1: j_global = 2 + j
                for x_idx, coeff in u_dict.items():
                    # x_idx 0,1,2,3 corresponds to (i,j): 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
                    i_local = x_idx // 2
                    j_local = x_idx % 2
                    i_global = i_local
                    j_global = 2 * a_block + j_local
                    U[i_global * m + j_global][col] += coeff
                
                # V: B indices
                # B local index (k, l) -> global B index k*p + l = k*6 + l
                # For a_block=0: k_global = k (0 or 1), l_global = 2*t + l
                # For a_block=1: k_global = 2 + k, l_global = 2*t + l
                for y_idx, coeff in v_dict.items():
                    k_local = y_idx // 2
                    l_local = y_idx % 2
                    k_global = 2 * a_block + k_local
                    l_global = 2 * t + l_local
                    V[k_global * p + l_global][col] += coeff
                
                # W: C indices
                # C local index (i, s) -> global W index (2*t + s)*n + i = (2t+s)*2 + i
                # (using trace convention W[k*n+i, r] where k is col, i is row)
                for z_idx, coeff in w_dict.items():
                    i_local = z_idx // 2
                    s_local = z_idx % 2
                    k_global = 2 * t + s_local  # column of C
                    i_global = i_local          # row of C
                    W[k_global * n + i_global][col] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
