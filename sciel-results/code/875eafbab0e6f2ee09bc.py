"""Rewritten program for matrix multiplication 2x4 by 4x6 with Strassen block decomposition."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    # We multiply A (2x4) by B (4x6) to get C (2x6).
    # Index convention:
    #   A: row i in {0,1}, col j in {0,1,2,3} -> U index i*4 + j
    #   B: row j in {0,1,2,3}, col k in {0..5} -> V index j*6 + k
    #   C: row i in {0,1}, col k in {0..5} -> W index k*2 + i (official trace convention)
    #
    # Split A into two 2x2 column blocks: A_s for s in {0,1}, cols [2s, 2s+1]
    # Split B into three 2x2 column blocks: B_g for g in {0,1,2}, cols [2g, 2g+1]
    #   B rows: top half j in {0,1} (call it B_top), bottom half j in {2,3} (B_bot)
    #
    # For each s (A block row-pair) and g (C column block), we compute a 2x2 product
    # C_sg[i, k] where i in {0,1} (A row), k in {0,1} (within the 2-col block of C).
    #
    # The product C_sg = A_s @ B_{g, s-block}
    # where A_s is 2x2 (rows 0,1 of A; cols 2s,2s+1)
    # and B_{g, s-block} is 2x2: rows 2s, 2s+1 of B; cols 2g, 2g+1.
    #
    # Strassen for 2x2: M1..M7 as usual.
    # Total rank = 3 (g values) * 2 (s values) * 7 (Strassen terms) = 42.

    n, m, p = DIMENSIONS  # 2, 4, 6

    # We'll build U, V, W as lists of lists.
    rank = 42
    U = [[0] * rank for _ in range(n * m)]   # 8 x 42
    V = [[0] * rank for _ in range(m * p)]   # 24 x 42
    W = [[0] * rank for _ in range(p * n)]   # 12 x 42

    # Strassen definitions for a 2x2 block product:
    # A_s = [[a00, a01], [a10, a11]]  (a_ij = A[i, 2s+j])
    # B_blk = [[b00, b01], [b10, b11]] (b_jk = B[2s+j, 2g+k])
    #
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11) * b00
    # M3 = a00 * (b01 - b11)
    # M4 = a11 * (b10 - b00)
    # M5 = (a00 + a01) * b11
    # M6 = (a10 - a00) * (b00 + b01)
    # M7 = (a01 - a11) * (b10 + b11)
    #
    # C_sg[0,0] = M1 - M4 + M5 + M7
    # C_sg[0,1] = M3 + M4
    # C_sg[1,0] = M2 + M5
    # C_sg[1,1] = M1 - M3 + M2 - M6

    col = 0
    for g in range(3):       # C column block
        for s in range(2):   # A/B row block
            for term in range(7):  # M1..M7
                # U coefficients (A side)
                if term == 0:   # M1: (a00 + a11)
                    U[0 * 4 + 2 * s][col] += 1
                    U[1 * 4 + (2 * s + 1)][col] += 1
                elif term == 1: # M2: (a10 + a11)
                    U[1 * 4 + 2 * s][col] += 1
                    U[1 * 4 + (2 * s + 1)][col] += 1
                elif term == 2: # M3: a00
                    U[0 * 4 + 2 * s][col] += 1
                elif term == 3: # M4: a11
                    U[1 * 4 + (2 * s + 1)][col] += 1
                elif term == 4: # M5: (a00 + a01)
                    U[0 * 4 + 2 * s][col] += 1
                    U[0 * 4 + (2 * s + 1)][col] += 1
                elif term == 5: # M6: (a10 - a00)
                    U[1 * 4 + 2 * s][col] += 1
                    U[0 * 4 + 2 * s][col] += -1
                elif term == 6: # M7: (a01 - a11)
                    U[0 * 4 + (2 * s + 1)][col] += 1
                    U[1 * 4 + (2 * s + 1)][col] += -1

                # V coefficients (B side)
                if term == 0:   # M1: (b00 + b11)
                    V[(2 * s) * 6 + 2 * g][col] += 1
                    V[(2 * s + 1) * 6 + (2 * g + 1)][col] += 1
                elif term == 1: # M2: b00
                    V[(2 * s) * 6 + 2 * g][col] += 1
                elif term == 2: # M3: (b01 - b11)
                    V[(2 * s) * 6 + (2 * g + 1)][col] += 1
                    V[(2 * s + 1) * 6 + (2 * g + 1)][col] += -1
                elif term == 3: # M4: (b10 - b00)
                    V[(2 * s + 1) * 6 + 2 * g][col] += 1
                    V[(2 * s) * 6 + 2 * g][col] += -1
                elif term == 4: # M5: b11
                    V[(2 * s + 1) * 6 + (2 * g + 1)][col] += 1
                elif term == 5: # M6: (b00 + b01)
                    V[(2 * s) * 6 + 2 * g][col] += 1
                    V[(2 * s) * 6 + (2 * g + 1)][col] += 1
                elif term == 6: # M7: (b10 + b11)
                    V[(2 * s + 1) * 6 + 2 * g][col] += 1
                    V[(2 * s + 1) * 6 + (2 * g + 1)][col] += 1

                # W coefficients (C side)
                # C_sg[i, k] where i in {0,1}, k in {0,1}
                # Global C col = 2*g + k, global C row = i
                # W index = (2*g + k) * 2 + i
                if term == 0:   # M1: contributes to C[0,0] (+1) and C[1,1] (+1)
                    W[(2 * g + 0) * 2 + 0][col] += 1
                    W[(2 * g + 1) * 2 + 1][col] += 1
                elif term == 1: # M2: contributes to C[1,0] (+1) and C[1,1] (+1)
                    W[(2 * g + 0) * 2 + 1][col] += 1
                    W[(2 * g + 1) * 2 + 1][col] += 1
                elif term == 2: # M3: contributes to C[0,1] (+1) and C[1,1] (-1)
                    W[(2 * g + 1) * 2 + 0][col] += 1
                    W[(2 * g + 1) * 2 + 1][col] += -1
                elif term == 3: # M4: contributes to C[0,0] (-1) and C[0,1] (+1)
                    W[(2 * g + 0) * 2 + 0][col] += -1
                    W[(2 * g + 1) * 2 + 0][col] += 1
                elif term == 4: # M5: contributes to C[0,0] (+1) and C[1,0] (+1)
                    W[(2 * g + 0) * 2 + 0][col] += 1
                    W[(2 * g + 0) * 2 + 1][col] += 1
                elif term == 5: # M6: contributes to C[1,1] (-1)
                    W[(2 * g + 1) * 2 + 1][col] += -1
                elif term == 6: # M7: contributes to C[0,0] (+1)
                    W[(2 * g + 0) * 2 + 0][col] += 1

                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
