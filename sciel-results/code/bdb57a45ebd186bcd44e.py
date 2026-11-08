import time

def construct():
    start = time.time()
    deadline = start + 110  # safety margin before 120s

    n, m, p = 6, 6, 6
    
    # Strassen algorithm for 6x6 matrices using 2x2 blocks of size 3x3
    # 7 multiplications of 3x3 matrices
    # Each 3x3 multiplication is done naively (rank 27)
    # Total rank = 7 * 27 = 189
    
    total_rank = 189
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a naive 3x3 multiplication term
    # A_block is at (bi, bj), B_block is at (bk, bl)
    # C_block is at (bi, bl)
    # Local indices i, j, k, l in 0..2
    # A[bi*3+i, bj*3+j] * B[bk*3+k, bl*3+l] -> C[bi*3+i, bl*3+l]
    # Wait, C = A B. C[i, l] = sum_j A[i, j] B[j, l].
    # So block C[bk, bi] = sum_j A[bi, j] B[j, bk].
    # Global C index: (bi*3 + i, bk*3 + l).
    # Global A index: (bi*3 + i, j*3 + j_local).
    # Global B index: (j*3 + k_local, bk*3 + l).
    
    def add_naive_3x3(bi, bj, bk, bl, sign_a, sign_b, sign_w):
        nonlocal col
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        # A index
                        r_u = (bi * 3 + i) * 6 + (bj * 3 + j)
                        # B index
                        r_v = (bj * 3 + k) * 6 + (bk * 3 + l) # Wait, B block is (j, bk)?
                        # In C[bi, bk] = sum_j A[bi, j] B[j, bk]
                        # The B block index is (j, bk).
                        # So r_v row is j*3 + k_local, col is bk*3 + l.
                        r_v = (bj * 3 + k) * 6 + (bk * 3 + l)
                        # C index
                        r_w = (bi * 3 + i) * 6 + (bk * 3 + l)
                        
                        U[r_u][col] += sign_a
                        V[r_v][col] += sign_b
                        W[r_w][col] += sign_w
                        col += 1
                        if time.time() > deadline:
                            return False
        return True

    # Strassen Steps
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # We need to decompose sums like (A11 + A22).
    # This means we add terms for A11 and A22 separately.
    
    # M1
    # (A11 + A22)(B11 + B22) = A11B11 + A11B22 + A22B11 + A22B22
    # Note: These are intermediate products. We must accumulate them into the final C.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # It is more efficient to expand the final C expressions directly?
    # No, Strassen is defined by the 7 Ms.
    # We will generate the 7 Ms, and then combine them.
    # But U, V, W are linear.
    # We can just add the contributions of each term in the expansion of the 7 Ms
    # multiplied by their coefficients in the final C.
    
    # Let's define the contributions of each (bi, bj) A-block and (bk, bl) B-block
    # to each C-block.
    
    # Instead of generating Ms and then combining, we can just iterate through
    # the 7 Strassen terms, and for each term, determine which C-blocks it contributes to
    # and with what sign.
    
    # Term M1: A_blocks: (0,0), (1,1). B_blocks: (0,0), (1,1).
    # M1 contributes to C(0,0) with +1, C(1,1) with +1.
    # But M1 is (A00+A11)(B00+B11).
    # This expands to 4 products.
    
    # This is getting complex to track signs manually.
    # Let's just generate the 7 Ms into temporary variables? No, U,V,W are global.
    # We can use the linearity.
    # For each of the 7 Strassen multiplications, we identify the A-blocks and B-blocks involved.
    # Then for each resulting C-block, we add the product with the appropriate sign.
    
    # Structure:
    # strassen_terms = [
    #   (A_blocks, B_blocks, C_contributions)
    # ]
    # A_blocks: list of (bi, bj, sign)
    # B_blocks: list of (bk, bl, sign)
    # C_contributions: list of (ci, ck, sign)
    
    # M1 = (A00 + A11)(B00 + B11)
    # A: [(0,0,1), (1,1,1)]
    # B: [(0,0,1), (1,1,1)]
    # C: C00 += M1, C11 += M1
    
    # M2 = (A10 + A11)B00
    # A: [(1,0,1), (1,1,1)]
    # B: [(0,0,1)]
    # C: C10 += M2, C11 -= M2
    
    # M3 = A00(B01 - B11)
    # A: [(0,0,1)]
    # B: [(0,1,1), (1,1,-1)]
    # C: C01 += M3, C11 += M3
    
    # M4 = A11(B10 - B00)
    # A: [(1,1,1)]
    # B: [(1,0,1), (0,0,-1)]
    # C: C00 += M4, C10 += M4
    
    # M5 = (A00 + A01)B11
    # A: [(0,0,1), (0,1,1)]
    # B: [(1,1,1)]
    # C: C00 -= M5, C01 += M5
    
    # M6 = (A10 - A00)(B00 + B01)
    # A: [(1,0,1), (0,0,-1)]
    # B: [(0,0,1), (0,1,1)]
    # C: C11 += M6
    
    # M7 = (A01 - A11)(B10 + B11)
    # A: [(0,1,1), (1,1,-1)]
    # B: [(1,0,1), (1,1,1)]
    # C: C00 += M7
    
    terms = [
        {
            'A': [(0,0,1), (1,1,1)],
            'B': [(0,0,1), (1,1,1)],
            'C': [(0,0,1), (1,1,1)]
        },
        {
            'A': [(1,0,1), (1,1,1)],
            'B': [(0,0,1)],
            'C': [(1,0,1), (1,1,-1)]
        },
        {
            'A': [(0,0,1)],
            'B': [(0,1,1), (1,1,-1)],
            'C': [(0,1,1), (1,1,1)]
        },
        {
            'A': [(1,1,1)],
            'B': [(1,0,1), (0,0,-1)],
            'C': [(0,0,1), (1,0,1)]
        },
        {
            'A': [(0,0,1), (0,1,1)],
            'B': [(1,1,1)],
            'C': [(0,0,-1), (0,1,1)]
        },
        {
            'A': [(1,0,1), (0,0,-1)],
            'B': [(0,0,1), (0,1,1)],
            'C': [(1,1,1)]
        },
        {
            'A': [(0,1,1), (1,1,-1)],
            'B': [(1,0,1), (1,1,1)],
            'C': [(0,0,1)]
        }
    ]
    
    for term in terms:
        for a_block in term['A']:
            bi, bj, sa = a_block
            for b_block in term['B']:
                bk, bl, sb = b_block
                # This product contributes to all C blocks in term['C']
                # with sign sa * sb * sc
                for c_block in term['C']:
                    ci, ck, sc = c_block
                    # Add naive 3x3 product of A[bi,bj] and B[bk,bl] to C[ci,ck]
                    # Note: B block index in Strassen is (row, col).
                    # In my add_naive_3x3, I used (bi, bj) for A and (bk, bl) for B?
                    # Let's re-verify add_naive_3x3 signature.
                    # It takes bi, bj (A block row, col) and bk, bl (B block row, col).
                    # And ci, ck (C block row, col).
                    # And signs.
                    
                    # The product is A[bi,bj] * B[bk,bl].
                    # This contributes to C[ci,ck] with sign sa*sb*sc.
                    
                    # We need to iterate local indices.
                    for i in range(3):
                        for j in range(3):
                            for k in range(3):
                                for l in range(3):
                                    r_u = (bi * 3 + i) * 6 + (bj * 3 + j)
                                    r_v = (bk * 3 + k) * 6 + (bl * 3 + l)
                                    r_w = (ci * 3 + i) * 6 + (ck * 3 + l)
                                    
                                    coeff = sa * sb * sc
                                    U[r_u][col] += coeff
                                    V[r_v][col] += coeff
                                    W[r_w][col] += coeff
                                    col += 1
                                    if time.time() > deadline:
                                        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
