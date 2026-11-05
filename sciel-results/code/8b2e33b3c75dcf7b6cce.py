"""New program: Strassen (4x4x4) + Schoolbook (4x4x2) decomposition, rank 60."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Rank 28 for 4x4x4 (Strassen outer, Schoolbook inner)
    # Rank 32 for 4x4x2 (Schoolbook)
    rank_4x4x4 = 28
    rank_4x4x2 = 32
    total_rank = rank_4x4x4 + rank_4x4x2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    def a_idx(i, j):
        return i * m + j

    def b_idx(j, k):
        return j * p + k

    def c_idx(k, i):
        return k * n + i

    # 1. 4x4x4 Part (k in 0..3)
    # Strassen products
    # A blocks: A00 (i:0-1, j:0-1), A01 (i:0-1, j:2-3), A10 (i:2-3, j:0-1), A11 (i:2-3, j:2-3)
    # B blocks: B00 (j:0-1, k:0-1), B01 (j:0-1, k:2-3), B10 (j:2-3, k:0-1), B11 (j:2-3, k:2-3)
    # C blocks: C00 (k:0-1, i:0-1), C01 (k:0-1, i:2-3), C10 (k:2-3, i:0-1), C11 (k:2-3, i:2-3)
    
    # Strassen definitions:
    # M1 = A00 * (B00 + B11)
    # M2 = (A10 + A00) * B00
    # M3 = (A01 - A11) * B11
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # Reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each M is a 2x2 matrix product.
    # We expand each M into 4 scalar products (schoolbook for 2x2x2).
    # For a 2x2 product P = A_block * B_block:
    # P[r, t] = sum_s A_block[r, s] * B_block[s, t]
    # The rank-1 tensors are for each (r, s, t) in {0,1}^3:
    # u: A_block[r, s]
    # v: B_block[s, t]
    # w: P[r, t] contribution to C
    
    # Define the 7 products with their A and B block coefficients
    # Format: (a_blocks, b_blocks)
    # a_blocks: list of (block_row, block_col, coeff)
    # b_blocks: list of (block_row, block_col, coeff)
    
    strassen_prods = [
        # M1
        ([(0, 0, 1)], [(0, 0, 1), (1, 1, 1)]),
        # M2
        ([(1, 0, 1), (0, 0, 1)], [(0, 0, 1)]),
        # M3
        ([(0, 1, 1), (1, 1, -1)], [(1, 1, 1)]),
        # M4
        ([(1, 1, 1)], [(1, 0, 1), (0, 0, -1)]),
        # M5
        ([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)]),
        # M6
        ([(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)]),
        # M7
        ([(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)]),
    ]
    
    # C reconstruction coefficients for each product
    # c_recon[c_block] = list of (prod_id, coeff)
    c_recon = {
        (0, 0): [(0, 1), (3, 1), (4, -1), (6, 1)],  # C00 = M1 + M4 - M5 + M7
        (0, 1): [(2, 1), (4, 1)],                   # C01 = M3 + M5
        (1, 0): [(1, 1), (3, 1)],                   # C10 = M2 + M4
        (1, 1): [(0, 1), (1, -1), (2, 1), (5, 1)],  # C11 = M1 - M2 + M3 + M6
    }
    
    col = 0
    for prod_id in range(7):
        a_terms, b_terms = strassen_prods[prod_id]
        
        # Expand 2x2 * 2x2 into 4 terms (r, s, t)
        for r in range(2):
            for s in range(2):
                for t in range(2):
                    # U contributions: A_block[r, s]
                    # For each (bi, bj, ca) in a_terms:
                    # Global i = 2*bi + r, Global j = 2*bj + s
                    for (bi, bj, ca) in a_terms:
                        gi = 2 * bi + r
                        gj = 2 * bj + s
                        U[a_idx(gi, gj)][col] += ca
                    
                    # V contributions: B_block[s, t]
                    # For each (bj, bk, cb) in b_terms:
                    # Global j = 2*bj + s, Global k = 2*bk + t
                    for (bj, bk, cb) in b_terms:
                        gj = 2 * bj + s
                        gk = 2 * bk + t
                        V[b_idx(gj, gk)][col] += cb
                    
                    # W contributions: C_block[r, t]
                    # The product M contributes to C blocks.
                    # The local indices (r, t) correspond to the position within the 2x2 C block.
                    # The C block is determined by the bi and bk of the A and B blocks?
                    # No, the C block is determined by the row of A block and col of B block?
                    # Actually, the Strassen product M is a 2x2 matrix.
                    # Its entry (r, t) contributes to the C block at position (r, t) in the 2x2 grid of C blocks?
                    # Yes.
                    # So the C block is (r, t) in the 2x2 grid?
                    # Wait, the C blocks are indexed by (c_row, c_col) in {0,1}^2.
                    # c_row corresponds to the row of the C matrix (i index high bit).
                    # c_col corresponds to the col of the C matrix (k index high bit).
                    # In the Strassen reconstruction, M is added to C blocks.
                    # M[r, t] adds to C block (r, t)?
                    # Let's check the indices.
                    # A block (bi, bj) has rows 2*bi..2*bi+1, cols 2*bj..2*bj+1.
                    # B block (bj, bk) has rows 2*bj..2*bj+1, cols 2*bk..2*bk+1.
                    # Product M = A_block * B_block has entries M[r, t].
                    # M[r, t] = sum_s A_block[r, s] B_block[s, t].
                    # This M[r, t] contributes to C[i, k] where i is in the row range of A block,
                    # and k is in the col range of B block.
                    # Specifically, i = 2*bi + r, k = 2*bk + t.
                    # So the C block is (bi, bk).
                    # The local indices within the C block are (r, t).
                    # So we need to find the coefficient for C block (bi, bk) from product prod_id.
                    
                    # However, bi and bk are not fixed for a given prod_id.
                    # a_terms contains multiple (bi, bj).
                    # b_terms contains multiple (bj, bk).
                    # The product is a sum of terms.
                    # The term (r, s, t) involves A_block(r, s) and B_block(s, t).
                    # The A_block comes from a_terms.
                    # The B_block comes from b_terms.
                    # But the C block depends on bi (from A) and bk (from B).
                    # This implies that for a single rank-1 tensor (r, s, t, prod_id),
                    # the U and V vectors are sums over the terms in a_terms and b_terms.
                    # But the W vector depends on which (bi, bk) pair is active.
                    # This means we cannot just sum the W coefficients for a single (r, s, t)
                    # if the (bi, bk) pairs are different.
                    #
                    # Wait, the Strassen product M is a single 2x2 matrix.
                    # M = A_expr * B_expr.
                    # A_expr is a linear combo of A blocks.
                    # B_expr is a linear combo of B blocks.
                    # M[r, t] is a scalar.
                    # The rank-1 tensor for this scalar is:
                    # u: A_expr[r, s] (vector over A indices)
                    # v: B_expr[s, t] (vector over B indices)
                    # w: 1 * (contribution to C)
                    #
                    # The contribution to C is:
                    # C[i, k] += coeff * M[r, t]
                    # where i = 2*bi + r, k = 2*bk + t.
                    # But bi and bk are not fixed.
                    # This suggests that the decomposition is not simply 28 rank-1 tensors
                    # if the blocks are mixed.
                    #
                    # Actually, the standard Strassen decomposition for 4x4x4
                    # with schoolbook inner products works as follows:
                    # Each of the 7 products is a 2x2 matrix multiplication.
                    # A 2x2 matrix multiplication has rank 4.
                    # So we have 7 * 4 = 28 rank-1 tensors.
                    # For each product p and each (r, s, t):
                    # u_p,r,s,t: Coefficients of A in A_expr_p[r, s]
                    # v_p,s,t: Coefficients of B in B_expr_p[s, t]
                    # w_p,r,t: Coefficients of C in the reconstruction.
                    #
                    # The issue is that w_p,r,t depends on the C block.
                    # The C block is determined by the high bits of i and k.
                    # i = 2*bi + r. So bi is the high bit of i.
                    # k = 2*bk + t. So bk is the high bit of k.
                    # The coefficient for C[i, k] is the sum of coefficients
                    # from all products p that contribute to C block (bi, bk).
                    #
                    # So for a fixed (p, r, s, t):
                    # u is fixed (based on a_terms of p).
                    # v is fixed (based on b_terms of p).
                    # w is: for each (bi, bk) in the support of a_terms x b_terms?
                    # No. The product M_p is a 2x2 matrix.
                    # M_p[r, t] is added to C blocks.
                    # Specifically, M_p[r, t] is added to C[i, k] where
                    # i is in the row range of the A block used in M_p?
                    # No, M_p is the result of multiplying A_expr_p and B_expr_p.
                    # A_expr_p is a 2x2 matrix.
                    # B_expr_p is a 2x2 matrix.
                    # M_p = A_expr_p * B_expr_p.
                    # M_p is a 2x2 matrix.
                    # This M_p is added to the C blocks.
                    # Which C blocks?
                    # The C blocks are 2x2 matrices.
                    # The addition is: C_block[bi, bk] += coeff * M_p.
                    # So M_p[r, t] is added to C_block[bi, bk][r, t].
                    # Which means C[2*bi+r, 2*bk+t] += coeff * M_p[r, t].
                    #
                    # So for a fixed (p, r, t), the W vector has non-zero entries
                    # at indices c_idx(2*bk+t, 2*bi+r) for all (bi, bk) such that
                    # the C block (bi, bk) receives a contribution from product p.
                    #
                    # Let's check the C reconstruction again.
                    # C00 = M1 + M4 - M5 + M7
                    # C01 = M3 + M5
                    # C10 = M2 + M4
                    # C11 = M1 - M2 + M3 + M6
                    #
                    # Here, C00 is the top-left 2x2 block of C.
                    # So C_block(0,0) receives M1, M4, -M5, M7.
                    # C_block(0,1) receives M3, M5.
                    # C_block(1,0) receives M2, M4.
                    # C_block(1,1) receives M1, -M2, M3, M6.
                    #
                    # So for product p, the set of C blocks it contributes to is fixed.
                    # For each such C block (bi, bk), the coefficient is fixed.
                    #
                    # So for a fixed (p, r, t):
                    # W vector has entries at c_idx(2*bk+t, 2*bi+r) for each (bi, bk)
                    # in the support of p's C contributions.
                    # The value is the coefficient for that C block.
                    
                    # Let's implement this.
                    
                    # Determine the C blocks for this product
                    # c_blocks_for_p: list of (bi, bk, coeff)
                    c_blocks_for_p = []
                    for (c_bi, c_bk) in c_recon.keys():
                        for (pi, cc) in c_recon[(c_bi, c_bk)]:
                            if pi == prod_id:
                                c_blocks_for_p.append((c_bi, c_bk, cc))
                                break
                        
                    # Fill W
                    for (c_bi, c_bk, cc) in c_blocks_for_p:
                        gi = 2 * c_bi + r
                        gk = 2 * c_bk + t
                        W[c_idx(gk, gi)][col] += cc
                    
                    col += 1
    
    # 2. 4x4x2 Part (k in 4..5)
    # Schoolbook
    # A is 4x4, B is 4x2 (cols 4,5), C is 4x2 (cols 4,5 of C? No, C is 4x6, so C cols 4,5)
    # C[k, i] for k in 4,5.
    # W index is k * 4 + i.
    
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                U[a_idx(i, j)][col] = 1
                V[b_idx(j, k)][col] = 1
                W[c_idx(k, i)][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
