"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 6
    # Block-schoolbook: partition into 2x2 blocks
    # A: 2x2 blocks of 2x2, B: 2x3 blocks of 2x2, C: 2x3 blocks of 2x2
    # C[ib, kb] = sum_{jb=0}^{1} A[ib, jb] @ B[jb, kb]
    # Each 2x2 @ 2x2 has 4 scalar products
    # Total terms: 2 * 3 * 2 * 4 = 48
    
    rank = 48
    U = [[0] * rank for _ in range(n * m)]  # (16, 48)
    V = [[0] * rank for _ in range(m * p)]  # (24, 48)
    W = [[0] * rank for _ in range(p * n)]  # (24, 48)
    
    col = 0
    for ib in range(2):       # i_block
        for kb in range(3):   # k_block
            for jb in range(2):  # j_block (summation index)
                for a_i in range(2):  # row within A block
                    for b_j in range(2):  # col within A block = row within B block
                        # A element: global row = ib*2 + a_i, global col = jb*2 + b_j
                        a_row = ib * 2 + a_i
                        a_col = jb * 2 + b_j
                        # B element: global row = jb*2 + b_j, global col = kb*2 + (a_i? no)
                        # B[jb, kb] is 2x2: B[jb*2 + b_j, kb*2 + a_i] where b_j is row in B block, a_i is col in B block
                        b_row = jb * 2 + b_j
                        b_col = kb * 2 + a_i
                        # C element: global row = ib*2 + a_i, global col = kb*2 + b_j
                        c_row = ib * 2 + a_i
                        c_col = kb * 2 + b_j
                        
                        # Trace convention: W[k*n + i, r] where output index = k*n + i
                        # Here k = c_col, i = c_row
                        w_idx = c_col * n + c_row
                        
                        U[a_row * m + a_col][col] = 1
                        V[b_row * p + b_col][col] = 1
                        W[w_idx][col] = 1
                        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
