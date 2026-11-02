"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a block Strassen decomposition with 2x2 blocks of 3x3 matrices.
    # Rank = R(M2) * R(M3) = 7 * 27 = 189.
    rank = 7 * 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen indices for 2x2 block multiplication
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Each Mi is a 3x3 matrix product.
    # We iterate over the 7 Strassen multiplications.
    # For each, we iterate over the 27 schoolbook terms of the 3x3 product.
    
    # Define the linear combinations for each Strassen step.
    # Each step defines X = sum alpha * A_block and Y = sum beta * B_block.
    # A_block indices: (block_row, block_col)
    # Global A index: i = block_row * 3 + r, j = block_col * 3 + c
    
    strassen_steps = [
        # (X_blocks, Y_blocks, W_blocks)
        # X_blocks: list of (block_idx, coeff)
        # Y_blocks: list of (block_idx, coeff)
        # W_blocks: list of (block_idx, coeff) for the result C
        (([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (0, 1, -1), (1, 0, 1), (1, 1, 1)])), # M1
        (([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)], [(1, 0, 1), (1, 1, 1)])), # M2
        (([(0, 0, 1)], [(0, 1, 1), (1, 1, -1)], [(0, 1, 1), (1, 1, 1)])), # M3
        (([(1, 1, 1)], [(1, 0, 1), (0, 0, -1)], [(0, 0, -1), (1, 0, 1)])), # M4
        (([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)], [(0, 0, -1), (0, 1, 1)])), # M5
        (([(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)], [(1, 1, 1)])), # M6
        (([(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)], [(0, 1, 1), (1, 1, 1)])), # M7
    ]
    
    # This structure is getting complex. Let's simplify.
    # We just need to generate the rank-1 terms.
    # For each Strassen step s (0..6):
    #   For each local r, c, k in 0..2:
    #     For each term in X_s: (bx, by, coeff_x)
    #     For each term in Y_s: (by, bz, coeff_y)
    #       Global A index: i = bx*3+r, j = by*3+c
    #       Global B index: k = by*3+c? No.
    #       The local product is X[r,c] * Y[c,k].
    #       X[r,c] comes from A block (bx, by). So A index is (bx*3+r, by*3+c).
    #       Y[c,k] comes from B block (by, bz). So B index is (by*3+c, bz*3+k).
    #       Wait, the inner index of X is 'c' (col of A block) and 'c' (row of B block).
    #       So the block col of A must match the block row of B.
    #       In Strassen, the blocks are aligned.
    #       So if X uses A block (bx, by), Y must use B block (by, bz).
    #       
    #     The result contributes to C block (bx, bz).
    #     Global C index: i = bx*3+r, l = bz*3+k.
    #     
    #     U index: i, j = bx*3+r, by*3+c
    #     V index: k, l = by*3+c, bz*3+k  -> V row is (by*3+c)*6 + (bz*3+k)
    #     W index: l, i = bz*3+k, bx*3+r  -> W row is (bz*3+k)*6 + (bx*3+r)
    #     
    #     Coeff: coeff_x * coeff_y
    
    col = 0
    for X_blocks, Y_blocks, W_blocks in strassen_steps:
        # X_blocks is a list of (bx, by, coeff)
        # Y_blocks is a list of (by, bz, coeff)
        # Note: 'by' in X must match 'by' in Y for a valid term.
        # In Strassen, the blocks are paired.
        
        for r in range(3):
            for c in range(3):
                for k in range(3):
                    # Iterate over all pairs of X and Y blocks
                    for bx, by_x, cx in X_blocks:
                        for by_y, bz, cy in Y_blocks:
                            if by_x != by_y:
                                continue
                            # Valid pair
                            coeff = cx * cy
                            if coeff == 0:
                                continue
                            
                            i = bx * 3 + r
                            j = by_x * 3 + c
                            k_b = by_y * 3 + c
                            l = bz * 3 + k
                            
                            # W block index is determined by the Strassen step's output.
                            # The W_blocks list tells us which C blocks this M_s contributes to.
                            # But in the tensor decomposition, each term goes to a specific C[i, l].
                            # The Strassen formula says M_s contributes to specific C blocks with specific signs.
                            # However, the tensor decomposition sums all terms.
                            # The signs are handled by the coefficients in X, Y, or W?
                            # In the standard Strassen, the signs are in the combination of M_s.
                            # Here, we are expanding M_s directly into C.
                            # So we need to know which C block (bx, bz) this term belongs to.
                            # The pair (bx, bz) is fixed by the X and Y blocks.
                            # But Strassen M_s might contribute to multiple C blocks?
                            # No, M_s is a single 3x3 matrix.
                            # It contributes to C blocks with coefficients.
                            # E.g., M1 contributes to C00 (+1), C01 (-1), C10 (+1), C11 (+1).
                            # So for a fixed (bx, bz) from X/Y, we need to check if it's a valid output block for this M_s.
                            # Actually, the pair (bx, bz) determines the C block.
                            # We just need to apply the correct coefficient from the Strassen output formula.
                            
                            # Let's define the output coefficients for each M_s.
                            # M1: C00:+1, C01:-1, C10:+1, C11:+1
                            # M2: C10:+1, C11:+1
                            # M3: C01:+1, C11:+1
                            # M4: C00:-1, C10:+1
                            # M5: C00:-1, C01:+1
                            # M6: C11:+1
                            # M7: C01:+1, C11:+1
                            
                            pass # Logic is too complex for inline.
                            
    # Simpler approach: Just return the schoolbook decomposition.
    # It is the only safe one I can write correctly without external data.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
