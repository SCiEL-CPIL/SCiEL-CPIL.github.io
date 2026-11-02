"""Program for 4x4x6 matrix multiplication tensor decomposition."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    
    # We'll build the decomposition in two parts:
    # Part 1: 4x4x4 (columns 0-3 of B) -> 4x4 output (rows 0-3, cols 0-3)
    # Part 2: 4x4x2 (columns 4-5 of B) -> 4x2 output (rows 0-3, cols 4-5)
    
    # For Part 1, use Strassen's algorithm on 2x2 blocks (rank 49)
    # For Part 2, use schoolbook (rank 32)
    
    rank = 49 + 32
    U = [[0] * rank for _ in range(n * m)]  # 16 x rank
    V = [[0] * rank for _ in range(m * p)]  # 24 x rank
    W = [[0] * rank for _ in range(p * n)]  # 24 x rank
    
    # Part 1: 4x4x4 multiplication using Strassen on 2x2 blocks
    # A is 4x4, split into 2x2 blocks: A00, A01, A10, A11
    # B is 4x4 (cols 0-3), split into 2x2 blocks: B00, B01, B10, B11
    # C is 4x4 (cols 0-3), split into 2x2 blocks: C00, C01, C10, C11
    
    # Strassen terms for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each of the 7 Strassen terms, we need to decompose the 2x2x2 multiplication
    # Each 2x2x2 multiplication has rank 7, so 7*7 = 49 total terms
    
    # Let me define the Strassen decomposition for a single 2x2x2:
    # Input A is 2x2: A[0,0], A[0,1], A[1,0], A[1,1] -> row index i*2+j where i,j in 0,1
    # Input B is 2x2: B[0,0], B[0,1], B[1,0], B[1,1] -> row index j*2+k where j,k in 0,1
    # Output C is 2x2: C[0,0], C[0,1], C[1,0], C[1,1] -> row index k*2+i where k,i in 0,1
    # (trace convention: W[k*n+i, r] but for local 2x2, n=2)
    
    # Strassen rank-7 decomposition for 2x2x2:
    # Term 1: (A00+A11)(B00+B11)
    #   U factors: A00, A11
    #   V factors: B00, B11
    #   W factors: C00, C11 (with signs)
    #   Actually, this is a single rank-1 term: (A00+A11) * (B00+B11) * (C00 + C11 contribution)
    #   But we need to decompose it into rank-1 terms.
    
    # Let me use the standard Strassen rank-7 decomposition:
    # The 7 terms are:
    # 1. (A00+A11)(B00+B11) -> contributes to C00, C11
    # 2. (A10+A11)B00 -> contributes to C10, C11
    # 3. A00(B01-B11) -> contributes to C01
    # 4. A11(B10-B00) -> contributes to C00, C10
    # 5. (A00+A01)B11 -> contributes to C01, C00
    # 6. (A10-A00)(B00+B01) -> contributes to C11
    # 7. (A01-A11)(B10+B11) -> contributes to C00
    
    # For each term, it's a single rank-1 tensor: U_factor * V_factor * W_factor
    # where U_factor is a vector in R^4 (for 2x2 A), V_factor in R^4 (for 2x2 B), W_factor in R^4 (for 2x2 C)
    
    # Let me define the 7 Strassen basis vectors:
    # U vectors (for A, 2x2 = 4 entries):
    u1 = [1, 0, 0, 1]  # A00 + A11
    u2 = [0, 0, 1, 1]  # A10 + A11
    u3 = [1, 0, 0, 0]  # A00
    u4 = [0, 0, 0, 1]  # A11
    u5 = [1, 1, 0, 0]  # A00 + A01
    u6 = [0, 0, 1, -1] # A10 - A00
    u7 = [0, 1, 0, -1] # A01 - A11
    
    # V vectors (for B, 2x2 = 4 entries):
    v1 = [1, 0, 0, 1]  # B00 + B11
    v2 = [1, 0, 0, 0]  # B00
    v3 = [0, 1, 0, -1] # B01 - B11
    v4 = [0, 0, 1, -1] # B10 - B00
    v5 = [0, 0, 0, 1]  # B11
    v6 = [1, 1, 0, 0]  # B00 + B01
    v7 = [0, 0, 1, 1]  # B10 + B11
    
    # W vectors (for C, 2x2 = 4 entries, trace convention k*2+i):
    # C00: i=0,k=0 -> index 0*2+0 = 0
    # C01: i=0,k=1 -> index 1*2+0 = 2
    # C10: i=1,k=0 -> index 0*2+1 = 1
    # C11: i=1,k=1 -> index 1*2+1 = 3
    w1 = [1, 0, 0, 1]  # C00 + C11
    w2 = [0, 1, 0, 1]  # C10 + C11
    w3 = [0, 0, 1, 0]  # C01
    w4 = [1, 1, 0, 0]  # C00 + C10
    w5 = [1, 0, 1, 0]  # C00 + C01
    w6 = [0, 0, 0, 1]  # C11
    w7 = [1, 0, 0, 0]  # C00
    
    # Now for the 4x4x4 part, we have 4 blocks of 2x2x2, each with 7 terms = 28 terms?
    # No, that's not right. Let me reconsider.
    
    # Actually, the 4x4x4 multiplication can be done by treating it as 2x2 block multiplication
    # where each block is 2x2. The block multiplication has the same structure as 2x2x2
    # but each "element" is a 2x2 matrix.
    
    # So we need to decompose the 2x2x2 block multiplication where each element is 2x2.
    # This means we need to decompose each of the 7 Strassen terms into 2x2x2 rank-1 terms.
    
    # For term 1: (A00+A11)(B00+B11) where A00,A11,B00,B11 are 2x2 matrices
    # This is (A00+A11) * (B00+B11) which is a 2x2 * 2x2 = 2x2 multiplication
    # We can decompose this using the rank-7 Strassen decomposition again!
    
    # So total rank = 7 (top level) * 7 (each 2x2x2) = 49
    
    # Let me build this recursively.
    
    # For the 4x4x4 part:
    # Block structure:
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # B = [[B00, B01], [B10, B11]] where each is 2x2 (cols 0-3)
    # C = [[C00, C01], [C10, C11]] where each is 2x2 (cols 0-3)
    
    # Strassen terms (each is a 2x2x2 multiplication):
    # Term 1: (A00+A11) * (B00+B11) -> contributes to C00 and C11
    # Term 2: (A10+A11) * B00 -> contributes to C10 and C11
    # Term 3: A00 * (B01-B11) -> contributes to C01
    # Term 4: A11 * (B10-B00) -> contributes to C00 and C10
    # Term 5: (A00+A01) * B11 -> contributes to C01 and C00
    # Term 6: (A10-A00) * (B00+B01) -> contributes to C11
    # Term 7: (A01-A11) * (B10+B11) -> contributes to C00
    
    # For each term, I need to decompose the 2x2x2 multiplication using rank 7.
    # But the "elements" here are 2x2 matrices, so I need to be careful with indices.
    
    # Let me define helper to add a Strassen term for a 2x2x2 where:
    # A_mat is a 2x2 matrix stored in global U with row offset
    # B_mat is a 2x2 matrix stored in global V with row offset  
    # C_mat is a 2x2 matrix stored in global W with row offset
    # And I add 7 rank-1 terms to U, V, W
    
    # Global index mapping for 4x4x4:
    # U: A[i,j] -> row i*4+j (i,j in 0..3)
    # V: B[j,k] -> row j*6+k (j in 0..3, k in 0..3) for cols 0-3
    # W: C[i,k] -> row k*4+i (k in 0..3, i in 0..3)
    
    # Block (bi, bj) of A (2x2 blocks): A[bi*2+di, bj*2+dj] for di,dj in 0,1
    #   -> U row: (bi*2+di)*4 + (bj*2+dj)
    # Block (bj, bk) of B: B[bj*2+dj, bk*2+dk] for dj,dk in 0,1
    #   -> V row: (bj*2+dj)*6 + (bk*2+dk)
    # Block (bi, bk) of C: C[bi*2+di, bk*2+dk] for di,dk in 0,1
    #   -> W row: (bk*2+dk)*4 + (bi*2+di)
    
    # Let me write a function to add a 2x2x2 Strassen decomposition
    def add_strassen_2x2x2(U, V, W, rank_offset, 
                           a_row_base, a_col_base, 
                           b_row_base, b_col_base, 
                           c_row_base, c_col_base,
                           a_vec, b_vec, c_vec):
        """Add 7 rank-1 terms for a 2x2x2 Strassen decomposition.
        a_vec, b_vec, c_vec are the 7 Strassen basis vectors (length 4 each).
        The 2x2 A matrix has entries at U[a_row_base + di*4 + (a_col_base + dj)] for di,dj in 0,1
        Wait, I need to be more careful.
        
        Let me define:
        A is 2x2: A[di,dj] for di,dj in 0,1
        Global U row for A[di,dj]: a_row_base + di*4 + (a_col_base + dj)
        
        B is 2x2: B[dj,dk] for dj,dk in 0,1  
        Global V row for B[dj,dk]: b_row_base + dj*6 + (b_col_base + dk)
        
        C is 2x2: C[di,dk] for di,dk in 0,1
        Global W row for C[di,dk]: c_row_base + dk*4 + (c_row_base_di + di)
        """
        # This is getting complex. Let me just hardcode the indices.
        pass
    
    # Let me take a simpler approach and just build the full 49-term decomposition explicitly.
    
    # I'll define the 49 terms by iterating over the 7 top-level Strassen terms
    # and for each, the 7 bottom-level Strassen terms.
    
    # Top-level term t (0-6):
    #   A_factor: combination of 2x2 blocks
    #   B_factor: combination of 2x2 blocks
    #   C_factor: combination of 2x2 blocks (with signs)
    
    # Let me define the top-level Strassen factors as lists of (block_i, block_j, coeff)
    # for A, B, C blocks.
    
    # A blocks: A00=(0,0), A01=(0,1), A10=(1,0), A11=(1,1)
    # B blocks: B00=(0,0), B01=(0,1), B10=(1,0), B11=(1,1)
    # C blocks: C00=(0,0), C01=(0,1), C10=(1,0), C11=(1,1)
    
    # Top-level term 1: (A00+A11)(B00+B11) -> C00 + C11
    #   A: {(0,0):1, (1,1):1}
    #   B: {(0,0):1, (1,1):1}
    #   C: {(0,0):1, (1,1):1}
    
    # Top-level term 2: (A10+A11)B00 -> C10 + C11
    #   A: {(1,0):1, (1,1):1}
    #   B: {(0,0):1}
    #   C: {(1,0):1, (1,1):1}
    
    # Top-level term 3: A00(B01-B11) -> C01
    #   A: {(0,0):1}
    #   B: {(0,1):1, (1,1):-1}
    #   C: {(0,1):1}
    
    # Top-level term 4: A11(B10-B00) -> C00 + C10
    #   A: {(1,1):1}
    #   B: {(1,0):1, (0,0):-1}
    #   C: {(0,0):1, (1,0):1}
    
    # Top-level term 5: (A00+A01)B11 -> C01 + C00
    #   A: {(0,0):1, (0,1):1}
    #   B: {(1,1):1}
    #   C: {(0,1):1, (0,0):1}
    
    # Top-level term 6: (A10-A00)(B00+B01) -> C11
    #   A: {(1,0):1, (0,0):-1}
    #   B: {(0,0):1, (0,1):1}
    #   C: {(1,1):1}
    
    # Top-level term 7: (A01-A11)(B10+B11) -> C00
    #   A: {(0,1):1, (1,1):-1}
    #   B: {(1,0):1, (1,1):1}
    #   C: {(0,0):1}
    
    # For each top-level term, I need to decompose the 2x2x2 where each "element" is a 2x2 matrix.
    # This means I need to use the bottom-level Strassen decomposition on the 2x2 matrices.
    
    # Let me define the bottom-level Strassen factors similarly.
    
    # Bottom-level term s (0-6):
    #   a_vec[s]: vector in R^4 for the 2x2 A matrix
    #   b_vec[s]: vector in R^4 for the 2x2 B matrix  
    #   c_vec[s]: vector in R^4 for the 2x2 C matrix
    
    # a_vec[0] = [1,0,0,1]  # A00+A11 (A[0,0]+A[1,1])
    # a_vec[1] = [0,0,1,1]  # A10+A11 (A[1,0]+A[1,1])
    # a_vec[2] = [1,0,0,0]  # A00
    # a_vec[3] = [0,0,0,1]  # A11
    # a_vec[4] = [1,1,0,0]  # A00+A01
    # a_vec[5] = [0,0,1,-1] # A10-A00
    # a_vec[6] = [0,1,0,-1] # A01-A11
    
    # b_vec[0] = [1,0,0,1]  # B00+B11
    # b_vec[1] = [1,0,0,0]  # B00
    # b_vec[2] = [0,1,0,-1] # B01-B11
    # b_vec[3] = [0,0,1,-1] # B10-B00
    # b_vec[4] = [0,0,0,1]  # B11
    # b_vec[5] = [1,1,0,0]  # B00+B01
    # b_vec[6] = [0,0,1,1]  # B10+B11
    
    # c_vec[0] = [1,0,0,1]  # C00+C11 (C[0,0]+C[1,1])
    # c_vec[1] = [0,1,0,1]  # C10+C11 (C[1,0]+C[1,1])
    # c_vec[2] = [0,0,1,0]  # C01
    # c_vec[3] = [1,1,0,0]  # C00+C10
    # c_vec[4] = [1,0,1,0]  # C00+C01
    # c_vec[5] = [0,0,0,1]  # C11
    # c_vec[6] = [1,0,0,0]  # C00
    
    # Now for the full 49-term construction:
    # For top-level term t (0-6) and bottom-level term s (0-6):
    #   The U contribution comes from top-level A_factor and bottom-level a_vec[s]
    #   The V contribution comes from top-level B_factor and bottom-level b_vec[s]
    #   The W contribution comes from top-level C_factor and bottom-level c_vec[s]
    
    # Let me define the top-level factors as dictionaries:
    top_A = [
        {(0,0):1, (1,1):1},  # t=0
        {(1,0):1, (1,1):1},  # t=1
        {(0,0):1},            # t=2
        {(1,1):1},            # t=3
        {(0,0):1, (0,1):1},  # t=4
        {(1,0):1, (0,0):-1}, # t=5
        {(0,1):1, (1,1):-1}, # t=6
    ]
    top_B = [
        {(0,0):1, (1,1):1},  # t=0
        {(0,0):1},            # t=1
        {(0,1):1, (1,1):-1}, # t=2
        {(1,0):1, (0,0):-1}, # t=3
        {(1,1):1},            # t=4
        {(0,0):1, (0,1):1},  # t=5
        {(1,0):1, (1,1):1},  # t=6
    ]
    top_C = [
        {(0,0):1, (1,1):1},  # t=0
        {(1,0):1, (1,1):1},  # t=1
        {(0,1):1},            # t=2
        {(0,0):1, (1,0):1},  # t=3
        {(0,1):1, (0,0):1},  # t=4
        {(1,1):1},            # t=5
        {(0,0):1},            # t=6
    ]
    
    bot_a = [
        [1,0,0,1],  # s=0
        [0,0,1,1],  # s=1
        [1,0,0,0],  # s=2
        [0,0,0,1],  # s=3
        [1,1,0,0],  # s=4
        [0,0,1,-1], # s=5
        [0,1,0,-1], # s=6
    ]
    bot_b = [
        [1,0,0,1],  # s=0
        [1,0,0,0],  # s=1
        [0,1,0,-1], # s=2
        [0,0,1,-1], # s=3
        [0,0,0,1],  # s=4
        [1,1,0,0],  # s=5
        [0,0,1,1],  # s=6
    ]
    bot_c = [
        [1,0,0,1],  # s=0
        [0,1,0,1],  # s=1
        [0,0,1,0],  # s=2
        [1,1,0,0],  # s=3
        [1,0,1,0],  # s=4
        [0,0,0,1],  # s=5
        [1,0,0,0],  # s=6
    ]
    
    # Now build the 49 terms:
    term_idx = 0
    for t in range(7):
        for s in range(7):
            col = term_idx
            
            # U contribution: for each (bi,bj) in top_A[t] with coeff ca,
            # and for each (di,dj) in bot_a[s] with coeff cb,
            # add ca*cb to U[(bi*2+di)*4 + (bj*2+dj), col]
            for (bi, bj), ca in top_A[t].items():
                for di in range(2):
                    for dj in range(2):
                        cb = bot_a[s][di*2+dj]
                        if cb != 0:
                            row = (bi*2+di)*4 + (bj*2+dj)
                            U[row][col] += ca * cb
            
            # V contribution: for each (bj,bk) in top_B[t] with coeff cb,
            # and for each (dj,dk) in bot_b[s] with coeff cc,
            # add cb*cc to V[(bj*2+dj)*6 + (bk*2+dk), col]
            for (bj, bk), cb in top_B[t].items():
                for dj in range(2):
                    for dk in range(2):
                        cc = bot_b[s][dj*2+dk]
                        if cc != 0:
                            row = (bj*2+dj)*6 + (bk*2+dk)
                            V[row][col] += cb * cc
            
            # W contribution: for each (bi,bk) in top_C[t] with coeff cc,
            # and for each (di,dk) in bot_c[s] with coeff cd,
            # add cc*cd to W[(bk*2+dk)*4 + (bi*2+di), col]
            for (bi, bk), cc in top_C[t].items():
                for di in range(2):
                    for dk in range(2):
                        cd = bot_c[s][di*2+dk]
                        if cd != 0:
                            row = (bk*2+dk)*4 + (bi*2+di)
                            W[row][col] += cc * cd
            
            term_idx += 1
    
    # Part 2: 4x4x2 multiplication (schoolbook, rank 32)
    # A is 4x4, B is 4x2 (cols 4-5), C is 4x2 (cols 4-5)
    # C[i,k] = sum_j A[i,j]*B[j,k] for i in 0..3, k in 4,5
    # W row: k*4+i for k in 4,5; i in 0..3
    
    base = 49
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                col = base + (i*4 + j)*2 + (k-4)
                U[i*4+j][col] = 1
                V[j*6+k][col] = 1
                W[k*4+i][col] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

