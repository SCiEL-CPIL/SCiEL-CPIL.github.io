def construct():
    n, m, p = 4, 4, 8
    rank = 98
    U = [[0] * rank for _ in range(n * m)]  # 16 x 98
    V = [[0] * rank for _ in range(m * p)]  # 32 x 98
    W = [[0] * rank for _ in range(p * n)]  # 32 x 98
    
    # 2x2x2 Strassen decomposition
    # For X (2x2), Y (2x2), P = X*Y (2x2)
    # X indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # Y indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # P indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    #
    # The 7 terms (u, v, w) where u in R^4, v in R^4, w in R^4:
    strassen_2x2 = [
        # N1 = (x00+x11)(y00+y11) -> p00 + p11
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        # N2 = (x10+x11)*y00 -> p10
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
        # N3 = x00*(y10-y11) -> p01
        ([1,0,0,0], [0,0,1,-1], [0,1,0,0]),
        # N4 = x11*(y01-y10) -> p01
        ([0,0,0,1], [0,1,-1,0], [0,1,0,0]),
        # N5 = (x00+x10)*y11 -> p10
        ([1,0,1,0], [0,0,0,1], [0,0,1,0]),
        # N6 = (x01-x00)*(y00+y10) -> p11
        ([-1,1,0,0], [1,0,1,0], [0,0,0,1]),
        # N7 = (x10-x11)*(y01+y11) -> p00
        ([0,0,1,-1], [0,1,0,1], [1,0,0,0]),
    ]
    
    # Now build 4x4x4 using nested Strassen
    # A (4x4) blocks: A00=A[0:2,0:2], A01=A[0:2,2:4], A10=A[2:4,0:2], A11=A[2:4,2:4]
    # B (4x4) blocks: B00, B01, B10, B11 similarly
    # C (4x4) blocks: C00, C01, C10, C11
    
    # Outer Strassen 7 products, each is 2x2 x 2x2
    # Each product uses inner Strassen (7 terms)
    # Total: 7*7 = 49 terms per 4x4x4
    
    # For 4x4x8, we have two 4x4x4 problems
    # Total: 49 + 49 = 98 terms
    
    # Helper: add a 2x2x2 Strassen decomposition at given offset
    # x_rows: the 4 global U-indices for the 2x2 X matrix (row-major)
    # y_rows: the 4 global V-indices for the 2x2 Y matrix (row-major)
    # p_rows: the 4 global W-indices for the 2x2 P matrix (row-major)
    # col_offset: starting column in U,V,W
    # Returns: next column offset
    
    def add_strassen_2x2(x_rows, y_rows, p_rows, col_offset):
        for u, v, w in strassen_2x2:
            for idx in range(4):
                if u[idx] != 0:
                    U[x_rows[idx]][col_offset] += u[idx]
                if v[idx] != 0:
                    V[y_rows[idx]][col_offset] += v[idx]
                if w[idx] != 0:
                    W[p_rows[idx]][col_offset] += w[idx]
            col_offset += 1
        return col_offset
    
    # For 4x4x4:
    # A[i,j] -> U index i*4+j (since m=4)
    # B[j,k] -> V index j*4+k (since p=4 for the sub-problem)
    # C[i,k] -> W index k*4+i (since n=4, and W is k*n+i)
    
    # Wait, I need to be careful. In the full problem:
    # U index: i*m + j = i*4 + j
    # V index: j*p + k = j*8 + k
    # W index: k*n + i = k*4 + i
    
    # For the left 4x4x4 (B columns 0-3):
    # A[i,j] -> U[i*4+j]
    # B[j,k] for k=0..3 -> V[j*8+k]
    # C[i,k] for k=0..3 -> W[k*4+i]
    
    # For the right 4x4x4 (B columns 4-7):
    # A[i,j] -> U[i*4+j]
    # B[j,k] for k=4..7 -> V[j*8+k]
    # C[i,k] for k=4..7 -> W[k*4+i]
    
    # Now, for each 4x4x4, the outer Strassen:
    # A blocks (each 2x2):
    #   A00: A[0:2,0:2] -> U indices: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5
    #   A01: A[0:2,2:4] -> U indices: 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7
    #   A10: A[2:4,0:2] -> U indices: 2*4+0=8, 2*4+1=9, 3*4+0=12, 3*4+1=13
    #   A11: A[2:4,2:4] -> U indices: 2*4+2=10, 2*4+3=11, 3*4+2=14, 3*4+3=15
    
    # B blocks (each 2x2), for left sub-problem (k=0..3):
    #   B00: B[0:2,0:2] -> V indices: 0*8+0=0, 0*8+1=1, 1*8+0=8, 1*8+1=9
    #   B01: B[0:2,2:4] -> V indices: 0*8+2=2, 0*8+3=3, 1*8+2=10, 1*8+3=11
    #   B10: B[2:4,0:2] -> V indices: 2*8+0=16, 2*8+1=17, 3*8+0=24, 3*8+1=25
    #   B11: B[2:4,2:4] -> V indices: 2*8+2=18, 2*8+3=19, 3*8+2=26, 3*8+3=27
    
    # For right sub-problem (k=4..7):
    #   B00: B[0:2,4:6] -> V indices: 0*8+4=4, 0*8+5=5, 1*8+4=12, 1*8+5=13
    #   B01: B[0:2,6:8] -> V indices: 0*8+6=6, 0*8+7=7, 1*8+6=14, 1*8+7=15
    #   B10: B[2:4,4:6] -> V indices: 2*8+4=20, 2*8+5=21, 3*8+4=28, 3*8+5=29
    #   B11: B[2:4,6:8] -> V indices: 2*8+6=22, 2*8+7=23, 3*8+6=30, 3*8+7=31
    
    # C blocks (each 2x2), for left sub-problem (k=0..3):
    #   C00: C[0:2,0:2] -> W indices: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5
    #   C01: C[0:2,2:4] -> W indices: 2*4+0=8, 2*4+1=9, 3*4+0=12, 3*4+1=13
    #   C10: C[2:4,0:2] -> W indices: 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7
    #   C11: C[2:4,2:4] -> W indices: 2*4+2=10, 2*4+3=11, 3*4+2=14, 3*4+3=15
    
    # For right sub-problem (k=4..7):
    #   C00: C[0:2,4:6] -> W indices: 4*4+0=16, 4*4+1=17, 5*4+0=20, 5*4+1=21
    #   C01: C[0:2,6:8] -> W indices: 6*4+0=24, 6*4+1=25, 7*4+0=28, 7*4+1=29
    #   C10: C[2:4,4:6] -> W indices: 4*4+2=18, 4*4+3=19, 5*4+2=22, 5*4+3=23
    #   C11: C[2:4,6:8] -> W indices: 6*4+2=26, 6*4+3=27, 7*4+2=30, 7*4+3=31
    
    # Outer Strassen products for 4x4x4:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B10-B11)
    # M4 = (A11)(B01-B10)
    # M5 = (A00+A10)(B11)
    # M6 = (A01-A00)(B00+B10)
    # M7 = (A10-A11)(B01+B11)
    
    # Each Mt is a 2x2 x 2x2 product. The X matrix is a linear combo of A-blocks,
    # Y matrix is a linear combo of B-blocks, and the result P contributes to C-blocks.
    
    # For the inner Strassen, I need to map the 2x2 X to global U indices,
    # 2x2 Y to global V indices, 2x2 P to global W indices.
    
    # This is getting complex. Let me define the block index mappings.
    
    # For a 2x2 block of A at position (bi,bj) where bi,bj in {0,1}:
    # The 4 entries are at A[2*bi+di, 2*bj+dj] for di,dj in {0,1}
    # U index: (2*bi+di)*4 + (2*bj+dj)
    
    # Similarly for B at (bj,bk):
    # V index: (2*bj+dj)*8 + (2*bk+dk)
    
    # And for C at (bi,bk):
    # W index: (2*bk+dk)*4 + (2*bi+di)
    
    # Let me define a function to get the 4 global indices for a 2x2 block.
    
    def a_block_indices(bi, bj):
        """U indices for A[2*bi:2*bi+2, 2*bj:2*bj+2] in row-major order"""
        return [(2*bi+di)*4 + (2*bj+dj) for di in range(2) for dj in range(2)]
    
    def b_block_indices_left(bj, bk):
        """V indices for B[2*bj:2*bj+2, 2*bk:2*bk+2] (left sub-problem, k=0..3)"""
        return [(2*bj+dj)*8 + (2*bk+dk) for dj in range(2) for dk in range(2)]
    
    def b_block_indices_right(bj, bk):
        """V indices for B[2*bj:2*bj+2, 4+2*bk:4+2*bk+2] (right sub-problem, k=4..7)"""
        return [(2*bj+dj)*8 + (4+2*bk+dk) for dj in range(2) for dk in range(2)]
    
    def c_block_indices_left(bi, bk):
        """W indices for C[2*bi:2*bi+2, 2*bk:2*bk+2] (left, k=0..3)"""
        return [(2*bk+dk)*4 + (2*bi+di) for di in range(2) for dk in range(2)]
    
    def c_block_indices_right(bi, bk):
        """W indices for C[2*bi:2*bi+2, 4+2*bk:4+2*bk+2] (right, k=4..7)"""
        return [(4+2*bk+dk)*4 + (2*bi+di) for di in range(2) for dk in range(2)]
    
    # Now, for each outer Strassen product Mt, the X is a linear combo of A-blocks,
    # Y is a linear combo of B-blocks, and P (the result) contributes to C-blocks.
    
    # But the inner Strassen operates on 2x2 matrices. The X in the inner Strassen
    # is the actual 2x2 matrix (which is a linear combo of A-blocks), and similarly for Y.
    
    # The key insight: the outer Strassen defines 7 products Mt = Xt * Yt where Xt and Yt
    # are specific 2x2 matrices (linear combos of the A and B blocks). Each Mt is then
    # decomposed with inner Strassen into 7 rank-1 terms.
    
    # For the tensor decomposition, I need to express each inner Strassen term's
    # u, v, w vectors in terms of the global indices.
    
    # The u vector for an inner Strassen term gives coefficients for the 4 entries of X.
    # But X itself is a linear combo of A-blocks. So the global U coefficients are
    # the product of the inner Strassen u-coefficients with the A-block combination coefficients.
    
    # This is getting very complex to do by hand. Let me just build it programmatically.
    
    # Approach: For each of the 7 outer Strassen products, determine:
    # - The 2x2 X matrix as a linear combo of the 4 A-blocks
    # - The 2x2 Y matrix as a linear combo of the 4 B-blocks
    # - The 2x2 P result as a linear combo of the 4 C-blocks
    # Then for each of the 7 inner Strassen terms, combine these.
    
    # Actually, I realize there's a cleaner way. The outer Strassen says:
    # C = f(M1, M2, ..., M7) where each Mt = Xt * Yt
    # And each Mt = g(N1, N2, ..., N7) where each Nt = xt * yt (inner Strassen)
    
    # So C = f(g(N1), ..., g(N7)) which is a linear function of the Nt's.
    # And each Nt = xt * yt where xt is a linear combo of X entries and yt of Y entries.
    # And X entries are linear combos of A-block entries, Y entries of B-block entries.
    
    # So ultimately, each rank-1 term in the full decomposition has:
    # - U coefficients: from the chain A -> X -> xt
    # - V coefficients: from the chain B -> Y -> yt
    # - W coefficients: from the chain C -> Mt -> Nt
    
    # Let me just build this up step by step in code.
    
    # Define the outer Strassen: for each of the 7 products, the X and Y are linear combos
    # of the 4 blocks. Represent as 4x4 matrices (coefficients for each block entry).
    
    # A-blocks: A00 (bi=0,bj=0), A01 (bi=0,bj=1), A10 (bi=1,bj=0), A11 (bi=1,bj=1)
    # Each block is 2x2, so 4 entries. Total 16 A entries.
    
    # For outer product t, X_t is a 2x2 matrix where each entry is a linear combo of A entries.
    # Similarly Y_t for B entries.
    
    # Let me define X_t as a 4x16 matrix (2x2 X entries, 16 A entries) and Y_t as 4x16.
    
    # Actually, let me think of it differently. The outer Strassen products:
    # M1 = (A00+A11)(B00+B11)
    #   X1 = A00+A11 (2x2), Y1 = B00+B11 (2x2)
    # M2 = (A10+A11)(B00)
    #   X2 = A10+A11, Y2 = B00
    # M3 = (A00)(B10-B11)
    #   X3 = A00, Y3 = B10-B11
    # M4 = (A11)(B01-B10)
    #   X4 = A11, Y4 = B01-B10
    # M5 = (A00+A10)(B11)
    #   X5 = A00+A10, Y5 = B11
    # M6 = (A01-A00)(B00+B10)
    #   X6 = A01-A00, Y6 = B00+B10
    # M7 = (A10-A11)(B01+B11)
    #   X7 = A10-A11, Y7 = B01+B11
    
    # And the C reconstruction:
    # C00 = M1 - M4 - M5 + M7
    # C01 = M3 + M4
    # C10 = M2 + M5
    # C11 = M1 - M2 - M3 + M6
    
    # For the inner Strassen of Mt = Xt * Yt:
    # The 7 inner terms Nt1,...,Nt7 are rank-1 terms for the 2x2x2 tensor.
    # Each Nts = xts * yts where xts is a linear combo of Xt's 4 entries,
    # yts is a linear combo of Yt's 4 entries.
    # And the contribution to Mt's 4 entries is given by the w vector.
    
    # So the global rank-1 term for (t,s) has:
    # U coefficients: for each A entry a, coefficient = sum over Xt entries of (inner u coeff) * (Xt entry's coefficient for a)
    # Similarly for V and W.
    
    # Let me code this up properly.
    
    # First, define the block structure for A:
    # A entry (i,j) where i,j in 0..3. Block (bi,bj) contains entries (2*bi+di, 2*bj+dj).
    # U index for A(i,j) = i*4 + j
    
    # For outer product t, Xt is a 2x2 matrix. Xt[di,dj] is a linear combo of A entries.
    # Let me define Xt[di,dj] = sum_{bi,bj} coeff_t[di,dj,bi,bj] * A_block(bi,bj)[di,dj]
    # where A_block(bi,bj)[di,dj] = A[2*bi+di, 2*bj+dj]
    
    # Actually, it's simpler: Xt[di,dj] is just a specific A entry or sum/diff of A entries.
    
    # Let me define for each outer product t, the mapping from Xt[di,dj] to A entries.
    
    # X1 = A00 + A11: X1[di,dj] = A[di,dj] + A[2+di,2+dj]
    # X2 = A10 + A11: X2[di,dj] = A[2+di,dj] + A[2+di,2+dj]
    # X3 = A00: X3[di,dj] = A[di,dj]
    # X4 = A11: X4[di,dj] = A[2+di,2+dj]
    # X5 = A00 + A10: X5[di,dj] = A[di,dj] + A[2+di,dj]
    # X6 = A01 - A00: X6[di,dj] = A[di,2+dj] - A[di,dj]
    # X7 = A10 - A11: X7[di,dj] = A[2+di,dj] - A[2+di,2+dj]
    
    # Similarly for Y (with B blocks).
    
    # For the inner Strassen, the u vector gives coefficients for Xt[0,0], Xt[0,1], Xt[1,0], Xt[1,1].
    # So the global U coefficient for A(i,j) is:
    # sum_{di,dj: Xt[di,dj] involves A(i,j)} (inner_u[di*2+dj]) * (coeff of A(i,j) in Xt[di,dj])
    
    # This is doable. Let me just code it up.
    
    # I'll define the X and Y mappings as functions that, given (di,dj), return a dict of {A_index: coeff}
    
    def x_mapping(t, di, dj):
        """Return dict {U_index: coeff} for Xt[di,dj]"""
        d = {}
        if t == 0:  # X1 = A00+A11
            d[di*4+dj] = 1
            d[(2+di)*4+(2+dj)] = 1
        elif t == 1:  # X2 = A10+A11
            d[(2+di)*4+dj] = 1
            d[(2+di)*4+(2+dj)] = 1
        elif t == 2:  # X3 = A00
            d[di*4+dj] = 1
        elif t == 3:  # X4 = A11
            d[(2+di)*4+(2+dj)] = 1
        elif t == 4:  # X5 = A00+A10
            d[di*4+dj] = 1
            d[(2+di)*4+dj] = 1
        elif t == 5:  # X6 = A01-A00
            d[di*4+(2+dj)] = 1
            d[di*4+dj] = -1
        elif t == 6:  # X7 = A10-A11
            d[(2+di)*4+dj] = 1
            d[(2+di)*4+(2+dj)] = -1
        return d
    
    def y_mapping_left(t, dj, dk):
        """Return dict {V_index: coeff} for Yt[dj,dk] (left sub-problem)"""
        d = {}
        if t == 0:  # Y1 = B00+B11
            d[dj*8+dk] = 1
            d[(2+dj)*8+(2+dk)] = 1
        elif t == 1:  # Y2 = B00
            d[dj*8+dk] = 1
        elif t == 2:  # Y3 = B10-B11
            d[(2+dj)*8+dk] = 1
            d[(2+dj)*8+(2+dk)] = -1
        elif t == 3:  # Y4 = B01-B10
            d[dj*8+(2+dk)] = 1
            d[(2+dj)*8+dk] = -1
        elif t == 4:  # Y5 = B11
            d[(2+dj)*8+(2+dk)] = 1
        elif t == 5:  # Y6 = B00+B10
            d[dj*8+dk] = 1
            d[(2+dj)*8+dk] = 1
        elif t == 6:  # Y7 = B01+B11
            d[dj*8+(2+dk)] = 1
            d[(2+dj)*8+(2+dk)] = 1
        return d
    
    def y_mapping_right(t, dj, dk):
        """Return dict {V_index: coeff} for Yt[dj,dk] (right sub-problem)"""
        d = {}
        if t == 0:  # Y1 = B00+B11 (B columns 4-7)
            d[dj*8+(4+dk)] = 1
            d[(2+dj)*8+(4+2+dk)] = 1
        elif t == 1:  # Y2 = B00
            d[dj*8+(4+dk)] = 1
        elif t == 2:  # Y3 = B10-B11
            d[(2+dj)*8+(4+dk)] = 1
            d[(2+dj)*8+(4+2+dk)] = -1
        elif t == 3:  # Y4 = B01-B10
            d[dj*8+(4+2+dk)] = 1
            d[(2+dj)*8+(4+dk)] = -1
        elif t == 4:  # Y5 = B11
            d[(2+dj)*8+(4+2+dk)] = 1
        elif t == 5:  # Y6 = B00+B10
            d[dj*8+(4+dk)] = 1
            d[(2+dj)*8+(4+dk)] = 1
        elif t == 6:  # Y7 = B01+B11
            d[dj*8+(4+2+dk)] = 1
            d[(2+dj)*8+(4+2+dk)] = 1
        return d
    
    def c_mapping_left(t, di, dk):
        """Return dict {W_index: coeff} for Mt[di,dk] contribution to C (left)"""
        # C00 = M1 - M4 - M5 + M7
        # C01 = M3 + M4
        # C10 = M2 + M5
        # C11 = M1 - M2 - M3 + M6
        # Mt[di,dk] is the (di,dk) entry of the Mt product.
        # C[2*bi+di, 2*bk+dk] gets contribution from Mt[di,dk] based on which C block it is.
        # C block (bi,bk) corresponds to C[2*bi:2*bi+2, 2*bk:2*bk+2]
        # W index for C[i,k] = k*4+i
        # So C[2*bi+di, 2*bk+dk] -> W index (2*bk+dk)*4 + (2*bi+di)
        
        # For Mt[di,dk], it contributes to C blocks:
        # If Mt is M1: contributes to C00 (bi=0,bk=0) and C11 (bi=1,bk=1)
        # etc.
        
        # Actually, Mt[di,dk] is the (di,dk) entry of the 2x2 result Mt.
        # Mt = C_block_combination. The 2x2 Mt corresponds to one of the C blocks? No.
        
        # Wait, I'm confusing myself. Let me reclarify.
        
        # In the outer Strassen for 4x4x4:
        # C (4x4) is divided into 4 blocks of 2x2: C00, C01, C10, C11
        # Each Mt is a 2x2 matrix (the product of two 2x2 matrices)
        # The C blocks are linear combos of the Mt's:
        #   C00 = M1 - M4 - M5 + M7
        #   C01 = M3 + M4
        #   C10 = M2 + M5
        #   C11 = M1 - M2 - M3 + M6
        
        # So Mt[di,dk] (the (di,dk) entry of Mt) contributes to:
        # - C00[di,dk] with coeff +1 if t==0 (M1)
        # - C11[di,dk] with coeff +1 if t==0 (M1)
        # - C00[di,dk] with coeff -1 if t==3 (M4)
        # - etc.
        
        d = {}
        coeffs = {
            0: {0: 1, 3: 1},   # M1 -> C00 (+1), C11 (+1)
            1: {1: 1},          # M2 -> C10 (+1)
            2: {1: 1},          # M3 -> C01 (+1)
            3: {0: -1, 1: 1},  # M4 -> C00 (-1), C01 (+1)
            4: {0: -1, 1: 1},  # M5 -> C00 (-1), C10 (+1)
            5: {3: 1},          # M6 -> C11 (+1)
            6: {0: 1},          # M7 -> C00 (+1)
        }
        # C block (bi,bk) -> W index for entry (di,dk): (2*bk+dk)*4 + (2*bi+di)
        for bk, bi in [(0,0), (0,1), (1,0), (1,1)]:
            # C block (bi,bk)
            block_id = bi*2+bk  # 0:C00, 1:C01, 2:C10, 3:C11
            if block_id in coeffs[t]:
                w_idx = (2*bk+dk)*4 + (2*bi+di)
                d[w_idx] = d.get(w_idx, 0) + coeffs[t][block_id]
        return d
    
    def c_mapping_right(t, di, dk):
        """Return dict {W_index: coeff} for Mt[di,dk] contribution to C (right)"""
        d = {}
        coeffs = {
            0: {0: 1, 3: 1},
            1: {1: 1},
            2: {1: 1},
            3: {0: -1, 1: 1},
            4: {0: -1, 1: 1},
            5: {3: 1},
            6: {0: 1},
        }
        for bk, bi in [(0,0), (0,1), (1,0), (1,1)]:
            block_id = bi*2+bk
            if block_id in coeffs[t]:
                w_idx = (4+2*bk+dk)*4 + (2*bi+di)
                d[w_idx] = d.get(w_idx, 0) + coeffs[t][block_id]
        return d
    
    # Now build the full decomposition
    col = 0
    for side in ['left', 'right']:
        for t in range(7):  # outer Strassen product
            for s in range(7):  # inner Strassen term
                u_vec, v_vec, w_vec = strassen_2x2[s]
                # u_vec: coefficients for Xt[0,0], Xt[0,1], Xt[1,0], Xt[1,1]
                # v_vec: coefficients for Yt[0,0], Yt[0,1], Yt[1,0], Yt[1,1]
                # w_vec: coefficients for Mt[0,0], Mt[0,1], Mt[1,0], Mt[1,1]
                
                # Build global U coefficients
                for di in range(2):
                    for dj in range(2):
                        if u_vec[di*2+dj] != 0:
                            x_map = x_mapping(t, di, dj)
                            for u_idx, coeff in x_map.items():
                                U[u_idx][col] += u_vec[di*2+dj] * coeff
                
                # Build global V coefficients
                if side == 'left':
                    for dj in range(2):
                        for dk in range(2):
                            if v_vec[dj*2+dk] != 0:
                                y_map = y_mapping_left(t, dj, dk)
                                for v_idx, coeff in y_map.items():
                                    V[v_idx][col] += v_vec[dj*2+dk] * coeff
                else:
                    for dj in range(2):
                        for dk in range(2):
                            if v_vec[dj*2+dk] != 0:
                                y_map = y_mapping_right(t, dj, dk)
                                for v_idx, coeff in y_map.items():
                                    V[v_idx][col] += v_vec[dj*2+dk] * coeff
                
                # Build global W coefficients
                if side == 'left':
                    for di in range(2):
                        for dk in range(2):
                            if w_vec[di*2+dk] != 0:
                                c_map = c_mapping_left(t, di, dk)
                                for w_idx, coeff in c_map.items():
                                    W[w_idx][col] += w_vec[di*2+dk] * coeff
                else:
                    for di in range(2):
                        for dk in range(2):
                            if w_vec[di*2+dk] != 0:
                                c_map = c_mapping_right(t, di, dk)
                                for w_idx, coeff in c_map.items():
                                    W[w_idx][col] += w_vec[di*2+dk] * coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
