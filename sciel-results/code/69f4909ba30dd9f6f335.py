"""Program for matrix multiplication 4x4x6 with reduced rank decomposition."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # We'll decompose into 4x4x4 (rank 49) + 4x4x2 (rank 32) = 81 total
    # But let's try to do better. Use Strassen for 4x4x4.
    
    # Actually, let me use a simpler but still improved approach:
    # For 4x4x4: use 7-term Strassen recursively -> each 2x2x2 block uses 7 mults
    # The 4x4x4 tensor can be decomposed with rank 49 using known algorithms.
    
    # For simplicity and correctness, let me use:
    # 4x4x4: rank 49 (Laderman-style)
    # 4x4x2: rank 16 (treat as A @ [b1, b2] where we compute column by column)
    
    # Actually, let me just use a clean Strassen decomposition for the 4x4x4 part.
    # Strassen for 2x2x2: rank 7
    # For 4x4x4: split into 2x2 blocks, each "multiply" of 2x2 blocks uses 7 mults
    # We have 7 Strassen multiplications, each involving 2x2 matrices
    # Each 2x2 "multiplication" in the Strassen formula corresponds to 4 scalar mults in schoolbook,
    # but with Strassen it's 7. So total = 7 * 4? No, that's not right.
    
    # Let me just implement Strassen for 2x2x2 and use it to build up.
    
    def strassen_2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition, rank 7."""
        # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
        # C = [[c00, c01], [c10, c11]]
        #
        # m1 = (a00+a11)(b00+b11)
        # m2 = (a10+a11)b00
        # m3 = a00(b01-b11)
        # m4 = a11(b10-b00)
        # m5 = (a00+a01)b11
        # m6 = (a10-a00)(b00+b01)
        # m7 = (a01-a11)(b10+b11)
        #
        # c00 = m1 + m4 - m5 + m7
        # c01 = m3 + m5
        # c10 = m2 + m4
        # c11 = m1 - m2 + m3 + m6
        
        rank = 7
        U = [[0]*rank for _ in range(4)]  # n*m = 2*2 = 4
        V = [[0]*rank for _ in range(4)]  # m*p = 2*2 = 4
        W = [[0]*rank for _ in range(4)]  # p*n = 2*2 = 4
        
        # U indices: i*2+j for i in [0,1], j in [0,1]
        # V indices: j*2+k for j in [0,1], k in [0,1]
        # W indices: k*2+i for k in [0,1], i in [0,1]
        
        # m1: (a00+a11)(b00+b11) -> contributes to c00 and c11
        # U: a00 (+1), a11 (+1)
        U[0][0] = 1  # a00
        U[3][0] = 1  # a11
        V[0][0] = 1  # b00
        V[3][0] = 1  # b11
        W[0][0] = 1  # c00 (k=0,i=0)
        W[3][0] = 1  # c11 (k=1,i=1)
        
        # m2: (a10+a11)b00 -> contributes to c10 and c11
        U[2][1] = 1  # a10
        U[3][1] = 1  # a11
        V[0][1] = 1  # b00
        W[2][1] = 1  # c10 (k=1,i=0)
        W[3][1] = -1  # c11 (k=1,i=1)
        
        # m3: a00(b01-b11) -> contributes to c01 and c11
        U[0][2] = 1  # a00
        V[1][2] = 1  # b01
        V[3][2] = -1  # b11
        W[1][2] = 1  # c01 (k=0,i=1)
        W[3][2] = 1  # c11 (k=1,i=1)
        
        # m4: a11(b10-b00) -> contributes to c00 and c10
        U[3][3] = 1  # a11
        V[2][3] = 1  # b10
        V[0][3] = -1  # b00
        W[0][3] = 1  # c00 (k=0,i=0)
        W[2][3] = 1  # c10 (k=1,i=0)
        
        # m5: (a00+a01)b11 -> contributes to c00 and c01
        U[0][4] = 1  # a00
        U[1][4] = 1  # a01
        V[3][4] = 1  # b11
        W[0][4] = -1  # c00 (k=0,i=0)
        W[1][4] = 1  # c01 (k=0,i=1)
        
        # m6: (a10-a00)(b00+b01) -> contributes to c11
        U[2][5] = 1  # a10
        U[0][5] = -1  # a00
        V[0][5] = 1  # b00
        V[1][5] = 1  # b01
        W[3][5] = 1  # c11 (k=1,i=1)
        
        # m7: (a01-a11)(b10+b11) -> contributes to c00
        U[1][6] = 1  # a01
        U[3][6] = -1  # a11
        V[2][6] = 1  # b10
        V[3][6] = 1  # b11
        W[0][6] = 1  # c00 (k=0,i=0)
        
        return U, V, W
    
    # For 4x4x4, I'll use a different approach: just use schoolbook for now to ensure correctness,
    # but with a smarter decomposition. Actually, let me use the fact that 4x4x4 can be done
    # with rank 49 using known algorithms. For now, let me use rank 64 (schoolbook) for 4x4x4
    # and rank 16 for 4x4x2, total 80. That's better than 96.
    
    # Actually, let me just do schoolbook for both but see if I can improve.
    # 4x4x4 schoolbook: rank 64
    # 4x4x2 schoolbook: rank 32
    # Total: 96 (same as current)
    
    # Let me try: 4x4x4 with rank 49, 4x4x2 with rank 16 = 65 total
    
    # For 4x4x2: A(4x4) @ B(4x2) = C(4x2)
    # This is equivalent to computing A@[b0] and A@[b1] where b0,b1 are columns
    # Each is a 4x4x1 problem. Schoolbook rank for 4x4x1 is 16.
    # Total for 4x4x2: 32 if independent, but we can do it in 16 if we share the U part?
    # No, because B is different. Let's just use rank 32 for simplicity, or find a better way.
    
    # Let me use rank 49 for 4x4x4 and rank 32 for 4x4x2 = 81 total.
    
    # For 4x4x4 with rank 49, I need a valid decomposition. Let me use Laderman's algorithm.
    # Laderman's 4x4 algorithm has rank 49.
    
    # This is complex to implement from scratch. Let me use a simpler approach:
    # Use Strassen recursively. 4x4x4 can be decomposed using 7 multiplications of 2x2x2,
    # but each "2x2x2 multiplication" in the Strassen formula for the 4x4 case actually
    # requires 4 scalar multiplications if done naively, or 7 if we use Strassen for those too.
    # 
    # The standard recursive Strassen for 4x4x4:
    # Split A, B into 2x2 blocks of 2x2 matrices.
    # Use 7 Strassen multiplications, each of which is a 2x2x2 matrix multiplication.
    # Each 2x2x2 matrix multiplication can be done with rank 7 (Strassen) or 4 (schoolbook).
    # If we use Strassen for the inner level: 7 * 7 = 49. That's it!
    
    # So 4x4x4 rank = 49 using two-level Strassen.
    
    # Now I need to construct U, V, W for the two-level Strassen 4x4x4.
    
    # Level 1: Strassen for 4x4 (blocks are 2x2)
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bj is 2x2
    # C = [[C00, C01], [C10, C11]]
    
    # m1 = (A00+A11)(B00+B11)
    # m2 = (A10+A11)B00
    # m3 = A00(B01-B11)
    # m4 = A11(B10-B00)
    # m5 = (A00+A01)B11
    # m6 = (A10-A00)(B00+B01)
    # m7 = (A01-A11)(B10+B11)
    
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # Each mk is a 2x2 matrix product, done with Strassen (rank 7).
    
    # Total rank = 7 * 7 = 49
    
    # Now let me construct the full U, V, W for 4x4x4.
    # U indices: i*4+j for i in [0,3], j in [0,3] (A is 4x4)
    # V indices: j*4+k for j in [0,3], k in [0,3] (B is 4x4)
    # W indices: k*4+i for k in [0,3], i in [0,3] (C is 4x4)
    
    # This is getting complex. Let me just use rank 64 for 4x4x4 and rank 16 for 4x4x2.
    # 4x4x2: we can do this with rank 16 by treating it as A @ B where B is 4x2.
    # Actually, the minimum rank for 4x4x2 is 16 (it's isomorphic to two 4x4x1 problems that can be combined).
    # Wait no, 4x4x2 has dimension 4*4*2=32 in schoolbook. The tensor rank is at least 16.
    
    # Let me just go with a safe improvement: use Strassen for the 4x4x4 part.
    # I'll implement it carefully.
    
    # For time's sake, let me use a simpler but still improved approach:
    # Use schoolbook for everything but note that we can't easily beat 96 without complex algorithms.
    # 
    # Actually, let me try rank 72: 4x4x4 with rank 56 (4*14, using some optimization) + 4x4x2 with rank 16.
    # This is getting too speculative. Let me just return the schoolbook decomposition
    # but I know I need to do better.
    
    # Let me implement the two-level Strassen for 4x4x4 properly.
    
    def build_strassen_4x4():
        """Build U, V, W for 4x4x4 using two-level Strassen, rank 49."""
        rank = 49
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(16)]
        W = [[0]*rank for _ in range(16)]
        
        # First level Strassen coefficients (7 terms)
        # Each term has a sign and which blocks it involves
        
        # m1 = (A00+A11)(B00+B11)
        # m2 = (A10+A11)B00
        # m3 = A00(B01-B11)
        # m4 = A11(B10-B00)
        # m5 = (A00+A01)B11
        # m6 = (A10-A00)(B00+B01)
        # m7 = (A01-A11)(B10+B11)
        
        # For each first-level term t (0-6), we have a 2x2x2 Strassen decomposition.
        # The inner Strassen has 7 sub-terms (s=0-6).
        
        # Let's define the first-level structure:
        # term -> (a_blocks, b_blocks, c_blocks, signs)
        # where a_blocks is a list of (block_idx, coeff) for A, etc.
        
        # A blocks: 00=(0,0)-(1,1), 01=(0,2)-(1,3), 10=(2,0)-(3,1), 11=(2,2)-(3,3)
        # B blocks: 00=(0,0)-(1,1), 01=(0,2)-(1,3), 10=(2,0)-(3,1), 11=(2,2)-(3,3)
        # C blocks: 00=(0,0)-(1,1), 01=(0,2)-(1,3), 10=(2,0)-(3,1), 11=(2,2)-(3,3)
        
        # A block row,col: block 00 -> rows 0-1, cols 0-1; block 01 -> rows 0-1, cols 2-3
        # block 10 -> rows 2-3, cols 0-1; block 11 -> rows 2-3, cols 2-3
        # U index for A[i][j]: i*4+j
        # V index for B[i][j]: i*4+j
        # W index for C[i][j]: j*4+i (trace convention W[k*n+i] where k is col of C, i is row)
        
        # Let me define helper functions
        def a_idx(r, c):
            return r * 4 + c
        
        def b_idx(r, c):
            return r * 4 + c
        
        def c_idx(r, c):
            # W[k*n+i] where C has row i, col k. In standard C[i][k], W index is k*4+i
            return c * 4 + r
        
        # First level terms: (name, a_parts, b_parts, c_parts)
        # a_parts: list of (block_row, block_col, coeff)
        # b_parts: list of (block_row, block_col, coeff)
        # c_parts: list of (block_row, block_col, coeff)
        
        first_level = [
            # m1 = (A00+A11)(B00+B11)
            {
                'a': [((0,0), 1), ((1,1), 1)],
                'b': [((0,0), 1), ((1,1), 1)],
                'c': [((0,0), 1), ((1,1), 1)]  # c00 += m1, c11 += m1
            },
            # m2 = (A10+A11)B00
            {
                'a': [((1,0), 1), ((1,1), 1)],
                'b': [((0,0), 1)],
                'c': [((1,0), 1), ((1,1), -1)]  # c10 += m2, c11 -= m2
            },
            # m3 = A00(B01-B11)
            {
                'a': [((0,0), 1)],
                'b': [((0,1), 1), ((1,1), -1)],
                'c': [((0,1), 1), ((1,1), 1)]  # c01 += m3, c11 += m3
            },
            # m4 = A11(B10-B00)
            {
                'a': [((1,1), 1)],
                'b': [((1,0), 1), ((0,0), -1)],
                'c': [((0,0), 1), ((1,0), 1)]  # c00 += m4, c10 += m4
            },
            # m5 = (A00+A01)B11
            {
                'a': [((0,0), 1), ((0,1), 1)],
                'b': [((1,1), 1)],
                'c': [((0,0), -1), ((0,1), 1)]  # c00 -= m5, c01 += m5
            },
            # m6 = (A10-A00)(B00+B01)
            {
                'a': [((1,0), 1), ((0,0), -1)],
                'b': [((0,0), 1), ((0,1), 1)],
                'c': [((1,1), 1)]  # c11 += m6
            },
            # m7 = (A01-A11)(B10+B11)
            {
                'a': [((0,1), 1), ((1,1), -1)],
                'b': [((1,0), 1), ((1,1), 1)],
                'c': [((0,0), 1)]  # c00 += m7
            },
        ]
        
        # Inner Strassen (2x2x2) terms
        inner_strassen = [
            # m1 = (a00+a11)(b00+b11)
            {
                'a': [((0,0), 1), ((1,1), 1)],
                'b': [((0,0), 1), ((1,1), 1)],
                'c': [((0,0), 1), ((1,1), 1)]
            },
            # m2 = (a10+a11)b00
            {
                'a': [((1,0), 1), ((1,1), 1)],
                'b': [((0,0), 1)],
                'c': [((1,0), 1), ((1,1), -1)]
            },
            # m3 = a00(b01-b11)
            {
                'a': [((0,0), 1)],
                'b': [((0,1), 1), ((1,1), -1)],
                'c': [((0,1), 1), ((1,1), 1)]
            },
            # m4 = a11(b10-b00)
            {
                'a': [((1,1), 1)],
                'b': [((1,0), 1), ((0,0), -1)],
                'c': [((0,0), 1), ((1,0), 1)]
            },
            # m5 = (a00+a01)b11
            {
                'a': [((0,0), 1), ((0,1), 1)],
                'b': [((1,1), 1)],
                'c': [((0,0), -1), ((0,1), 1)]
            },
            # m6 = (a10-a00)(b00+b01)
            {
                'a': [((1,0), 1), ((0,0), -1)],
                'b': [((0,0), 1), ((0,1), 1)],
                'c': [((1,1), 1)]
            },
            # m7 = (a01-a11)(b10+b11)
            {
                'a': [((0,1), 1), ((1,1), -1)],
                'b': [((1,0), 1), ((1,1), 1)],
                'c': [((0,0), 1)]
            },
        ]
        
        for t_idx, term in enumerate(first_level):
            for s_idx, inner in enumerate(inner_strassen):
                col = t_idx * 7 + s_idx
                
                # Combine A parts
                for (br, bc), coeff_a in term['a']:
                    for (ir, ic), coeff_inner_a in inner['a']:
                        # Global row = br*2 + ir, global col = bc*2 + ic
                        gr = br * 2 + ir
                        gc = bc * 2 + ic
                        U[a_idx(gr, gc)][col] += coeff_a * coeff_inner_a
                
                # Combine B parts
                for (br, bc), coeff_b in term['b']:
                    for (ir, ic), coeff_inner_b in inner['b']:
                        gr = br * 2 + ir
                        gc = bc * 2 + ic
                        V[b_idx(gr, gc)][col] += coeff_b * coeff_inner_b
                
                # Combine C parts
                for (br, bc), coeff_c in term['c']:
                    for (ir, ic), coeff_inner_c in inner['c']:
                        gr = br * 2 + ir
                        gc = bc * 2 + ic
                        W[c_idx(gr, gc)][col] += coeff_c * coeff_inner_c
        
        return U, V, W
    
    U_444, V_444, W_444 = build_strassen_4x4()
    
    # Now for 4x4x2: A(4x4) @ B(4x2) = C(4x2)
    # Use schoolbook: rank = 4*4*2 = 32
    U_442 = [[0]*32 for _ in range(16)]
    V_442 = [[0]*32 for _ in range(8)]  # m*p = 4*2 = 8
    W_442 = [[0]*32 for _ in range(8)]  # p*n = 2*4 = 8
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(2):
                U_442[i*4+j][col] = 1
                V_442[j*2+k][col] = 1
                W_442[k*4+i][col] = 1
                col += 1
    
    # Combine: total rank = 49 + 32 = 81
    total_rank = 49 + 32
    
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(24)]  # m*p = 4*6 = 24
    W = [[0]*total_rank for _ in range(24)]  # p*n = 6*4 = 24
    
    # Copy 4x4x4 part (columns 0-3 of B, C)
    for r in range(16):
        for c in range(49):
            U[r][c] = U_444[r][c]
    for r in range(16):
        for c in range(49):
            V[r][c] = V_444[r][c]
    for r in range(16):
        for c in range(49):
            W[r][c] = W_444[r][c]
    
    # Copy 4x4x2 part (columns 4-5 of B, C)
    # V indices: for 4x4x2, V has shape (4*2, 32) = (8, 32)
    # Global V index for 4x4x6: V[j*6+k] for j in [0,3], k in [0,5]
    # For the 4x4x2 part, k is 4 or 5 (global), local k is 0 or 1
    # Local V index: j*2 + k_local
    # Global V index: j*6 + (k_local + 4)
    
    for r_local in range(8):
        j = r_local // 2
        k_local = r_local % 2
        r_global = j * 6 + (k_local + 4)
        for c in range(32):
            V[r_global][49 + c] = V_442[r_local][c]
    
    # W indices: for 4x4x2, W has shape (2*4, 32) = (8, 32)
    # Local W index: k_local*4 + i
    # Global W index: (k_local+4)*4 + i
    
    for r_local in range(8):
        k_local = r_local // 4
        i = r_local % 4
        r_global = (k_local + 4) * 4 + i
        for c in range(32):
            W[r_global][49 + c] = W_442[r_local][c]
    
    # U for 4x4x2 is same shape as 4x4x4 (16 x 32)
    for r in range(16):
        for c in range(32):
            U[r][49 + c] = U_442[r][c]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

