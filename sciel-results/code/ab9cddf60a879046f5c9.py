"""Matrix multiplication 6x6x6 via 2x2 Strassen block decomposition
with schoolbook rank-9 decomposition for each 3x3 block product.
Total rank: 7 * 9 = 63.
"""

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    bs = 3  # block size
    nb = 2   # number of blocks per dimension

    # Strassen 2x2 block level (7 products):
    # P1 = (A00+A11)(B00+B11) -> C00 += P1, C11 += P1
    # P2 = (A10+A11)(B00)     -> C10 += P2, C11 -= P2
    # P3 = A00(B01-B11)       -> C01 += P3, C11 += P3
    # P4 = A11(B10-B00)       -> C00 += P4, C10 += P4
    # P5 = (A00+A01)(B11)     -> C00 -= P5, C01 += P5
    # P6 = (A10-A00)(B00+B01) -> C11 += P6
    # P7 = (A01-A11)(B10+B11) -> C00 += P7
    #
    # Block indices: 0=A00, 1=A01, 2=A10, 3=A11
    # Block (bi): row offset = (bi // 2) * bs, col offset = (bi % 2) * bs

    strassen_terms = [
        # (A_block_coeffs, B_block_coeffs, C_block_coeffs)
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),       # P1
        ([(2,1),(3,1)], [(0,1)],           [(2,1),(3,-1)]),    # P2
        ([(0,1)],           [(1,1),(3,-1)],[(1,1),(3,1)]),     # P3
        ([(3,1)],           [(2,1),(0,-1)],[(0,1),(2,1)]),     # P4
        ([(0,1),(1,1)],     [(3,1)],       [(0,-1),(1,1)]),    # P5
        ([(2,1),(0,-1)],    [(0,1),(1,1)], [(3,1)]),           # P6
        ([(1,1),(3,-1)],    [(2,1),(3,1)], [(0,1)]),           # P7
    ]
    
    rank = 7 * 9
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for term_idx in range(7):
        a_comb, b_comb, c_comb = strassen_terms[term_idx]
        for i in range(bs):
            for k in range(bs):
                for j in range(bs):
                    # A contribution: sum of block coefficients * A[blk_row+i, blk_col+k]
                    for (blk_a, coeff_a) in a_comb:
                        row_a = (blk_a // 2) * bs
                        col_a = (blk_a % 2) * bs
                        u_idx = (row_a + i) * m + (col_a + k)
                        U[u_idx][col] += coeff_a
                    # B contribution: sum of block coefficients * B[blk_row+k, blk_col+j]
                    for (blk_b, coeff_b) in b_comb:
                        row_b = (blk_b // 2) * bs
                        col_b = (blk_b % 2) * bs
                        v_idx = (row_b + k) * p + (col_b + j)
                        V[v_idx][col] += coeff_b
                    # C contribution: sum of block coefficients * C[blk_row+i, blk_col+j]
                    for (blk_c, coeff_c) in c_comb:
                        row_c = (blk_c // 2) * bs
                        col_c = (blk_c % 2) * bs
                        w_idx = (col_c + j) * n + (row_c + i)
                        W[w_idx][col] += coeff_c
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
