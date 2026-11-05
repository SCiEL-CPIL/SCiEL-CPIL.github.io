"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-189 decomposition for 6x6 matrix multiplication using
    one level of Strassen (7 multiplications of 3x3 blocks)
    with each 3x3 multiplication done via schoolbook (rank 27).
    Total rank = 7 * 27 = 189.
    
    The key insight is that we can use Strassen at the top level
    to reduce from 36 scalar mults per block-pair to 7 block mults,
    each of which is a 3x3 multiplication requiring 27 scalar mults.
    """
    n = m = p = 6
    rank = 189
    
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    # Strassen decomposition for 6x6 = 2x2 blocks of 3x3
    # A = [[A00, A01], B = [[B00, B01], C = [[C00, C01],
    #     [A10, A11]]     [B10, B11]]     [C10, C11]]
    #
    # 7 Strassen products:
    # T1 = (A00+A11)(B00+B11)
    # T2 = (A10+A11)B00
    # T3 = A00(B01-B11)
    # T4 = A11(B10-B00)
    # T5 = (A00+A01)B11
    # T6 = (A10-A00)(B00+B01)
    # T7 = (A01-A11)(B10+B11)
    #
    # Result:
    # C00 = T1 + T4 - T5 + T7
    # C01 = T3 + T5
    # C10 = T2 + T4
    # C11 = T1 - T2 + T3 + T6
    
    # For each Strassen term, we need to build the U, V, W contributions.
    # Each Strassen term computes X * Y where X and Y are 3x3 linear combinations.
    # We decompose X*Y via schoolbook: for each (i,a,k), one scalar mult.
    
    # Define the 7 Strassen terms as (left_blocks, right_blocks)
    # left_blocks: dict mapping (block_row, block_col) -> coefficient for A
    # right_blocks: dict mapping (block_row, block_col) -> coefficient for B
    
    strassen_defs = [
        # T1: (A00+A11) * (B00+B11)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # T2: (A10+A11) * B00
        ({(1,0):1, (1,1):1}, {(0,0):1}),
        # T3: A00 * (B01-B11)
        ({(0,0):1}, {(0,1):1, (1,1):-1}),
        # T4: A11 * (B10-B00)
        ({(1,1):1}, {(1,0):1, (0,0):-1}),
        # T5: (A00+A01) * B11
        ({(0,0):1, (0,1):1}, {(1,1):1}),
        # T6: (A10-A00) * (B00+B01)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
        # T7: (A01-A11) * (B10+B11)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),
    ]
    
    # Which result blocks each Strassen term contributes to, and with what sign
    # C00 = T1 + T4 - T5 + T7  -> T1:+, T4:+, T5:-, T7:+
    # C01 = T3 + T5            -> T3:+, T5:+
    # C10 = T2 + T4            -> T2:+, T4:+
    # C11 = T1 - T2 + T3 + T6  -> T1:+, T2:-, T3:+, T6:+
    
    # For Strassen term t (0-indexed), which result blocks does it affect?
    # result_blocks[t] = list of (result_block_row, result_block_col, sign)
    result_blocks = [
        [(0,0,1), (1,1,1)],      # T1 -> C00(+), C11(+)
        [(1,0,1), (1,1,-1)],     # T2 -> C10(+), C11(-)
        [(0,1,1), (1,1,1)],      # T3 -> C01(+), C11(+)
        [(0,0,1), (1,0,1)],      # T4 -> C00(+), C10(+)
        [(0,0,-1), (0,1,1)],     # T5 -> C00(-), C01(+)
        [(1,1,1)],               # T6 -> C11(+)
        [(0,0,1)],               # T7 -> C00(+)
    ]
    
    col = 0
    for t_idx in range(7):
        left_blocks, right_blocks = strassen_defs[t_idx]
        
        # For this Strassen term, compute X*Y where:
        # X[i][a] = sum over (br,bc) in left_blocks: coeff * A[3*br+i][3*bc+a]
        # Y[a][k] = sum over (br,bc) in right_blocks: coeff * B[3*br+a][3*bc+k]
        # Z[i][k] = sum_a X[i][a] * Y[a][k]
        
        for i in range(3):
            for a in range(3):
                for k in range(3):
                    # U contribution: for each block in left_blocks
                    for (br, bc), coeff_l in left_blocks.items():
                        row_A = 3 * br + i
                        col_A = 3 * bc + a
                        U[row_A * 6 + col_A][col] += coeff_l
                    
                    # V contribution: for each block in right_blocks
                    for (br, bc), coeff_r in right_blocks.items():
                        row_B = 3 * br + a
                        col_B = 3 * bc + k
                        V[row_B * 6 + col_B][col] += coeff_r
                    
                    # W contribution: for each result block this term affects
                    for (rbr, rbc, sign) in result_blocks[t_idx]:
                        i_total = 3 * rbr + i
                        k_total = 3 * rbc + k
                        # W row index = k*n + i (trace convention)
                        W_row = k_total * 6 + i_total
                        W[W_row][col] += sign
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
