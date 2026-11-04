"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """Block-Strassen decomposition of 6x6 matrix multiplication.
    
    View 6x6 matrices as 3x3 grids of 2x2 blocks.
    Each 2x2 block multiplication uses Strassen's 7-term algorithm.
    Total rank: 27 block positions * 7 Strassen terms = 189.
    """
    n, m, p = DIMENSIONS
    # 3x3 grid of 2x2 blocks
    block_size = 2
    n_blocks = n // block_size  # 3
    m_blocks = m // block_size  # 3
    p_blocks = p // block_size  # 3
    
    # Strassen's 7 multiplication terms for 2x2 matrices
    # For X = [[x0,x1],[x2,x3]] * Y = [[y0,y1],[y2,y3]]:
    # m1 = (x0+x3)(y0+y3)
    # m2 = (x2+x3)y0
    # m3 = x0(y1-y3)
    # m4 = x3(y2-x0)
    # m5 = (x0+x1)y3
    # m6 = (x2-x0)(y0+y1)
    # m7 = (x1-x3)(y2+y3)
    
    # Each term is a product of two linear forms.
    # We represent each as (u_coeffs, v_coeffs, w_coeffs) where
    # u_coeffs has 4 entries (for the 2x2 block of A),
    # v_coeffs has 4 entries (for the 2x2 block of B),
    # w_coeffs has 4 entries (for the 2x2 block of C).
    
    # Strassen terms: (u_coeffs, v_coeffs, w_coeffs)
    # Indexing within 2x2 block: [0,1,2,3] = [[0,1],[2,3]]
    strassen_terms = [
        # m1 = (x0+x3)(y0+y3) -> contributes to z1(+1), z4(-1)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, -1]),
        # m2 = (x2+x3)y0 -> contributes to z3(+1), z4(-1)
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # m3 = x0(y1-y3) -> contributes to z2(+1), z4(+1)
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # m4 = x3(y2-x0) -> contributes to z1(+1), z3(+1)
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        # m5 = (x0+x1)y3 -> contributes to z1(-1), z2(+1)
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # m6 = (x2-x0)(y0+y1) -> contributes to z4(+1)
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # m7 = (x1-x3)(y2+y3) -> contributes to z1(+1)
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    rank = n_blocks * m_blocks * p_blocks * len(strassen_terms)
    # rank = 3 * 3 * 3 * 7 = 189
    
    U = [[0] * rank for _ in range(n * m)]  # (36, 189)
    V = [[0] * rank for _ in range(m * p)]  # (36, 189)
    W = [[0] * rank for _ in range(p * n)]  # (36, 189)
    
    col = 0
    for ib in range(n_blocks):      # block row of A
        for jb in range(m_blocks):  # block col of A (= block row of B)
            for kb in range(p_blocks):  # block col of B
                for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
                    # A block (ib, jb): 2x2 matrix at rows ib*2..ib*2+1, cols jb*2..jb*2+1
                    # B block (jb, kb): 2x2 matrix at rows jb*2..jb*2+1, cols kb*2..kb*2+1
                    # C block (ib, kb): 2x2 matrix at rows ib*2..ib*2+1, cols kb*2..kb*2+1
                    
                    for local_idx in range(4):
                        # Map local 2x2 index to global index
                        # local_idx: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
                        lr, lc = divmod(local_idx, 2)
                        
                        # U: row = ib*block_size + lr, col = jb*block_size + lc
                        u_row = ib * block_size + lr
                        u_col = jb * block_size + lc
                        U[u_row * m + u_col][col] = u_coeffs[local_idx]
                        
                        # V: row = jb*block_size + lr, col = kb*block_size + lc
                        v_row = jb * block_size + lr
                        v_col = kb * block_size + lc
                        V[v_row * p + v_col][col] = v_coeffs[local_idx]
                        
                        # W: row = kb*block_size + lr, col = ib*block_size + lc
                        # W follows trace convention: W[k*n+i, r]
                        # For C[ib,kb] block: row index in C is ib, col index is kb
                        # W index: k*n + i where k is col of C, i is row of C
                        # So W[(kb*block_size + lc) * n + (ib*block_size + lr)]
                        w_row = kb * block_size + lc
                        w_col = ib * block_size + lr
                        W[w_row * n + w_col][col] = w_coeffs[local_idx]
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
