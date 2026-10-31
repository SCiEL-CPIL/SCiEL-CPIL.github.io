import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to get global index for A/B entry
    # A is 4x4. Block (i,j) entry (u,v) -> global row 2i+u, col 2j+v
    # Index = row * 4 + col
    
    # Outer Strassen block definitions for X (from A) and Y (from B)
    # l in 0..6
    # X is 2x2, Y is 2x2
    # We define X[u][v] and Y[u][v] as lists of (block_idx, coeff)
    # block_idx is 0..3 corresponding to A00, A01, A10, A11
    
    # l=0: X = A00 + A11, Y = B00 + B11
    # l=1: X = A10 + A11, Y = B00
    # l=2: X = A00,       Y = B01 - B11
    # l=3: X = A11,       Y = B10 - B00
    # l=4: X = A00 + A01, Y = B11
    # l=5: X = A10 - A00, Y = B00 + B01
    # l=6: X = A01 - A11, Y = B10 + B11

    X_defs = [
        [(0, 1), (3, 1)],       # l=0
        [(2, 1), (3, 1)],       # l=1
        [(0, 1)],               # l=2
        [(3, 1)],               # l=3
        [(0, 1), (1, 1)],       # l=4
        [(2, 1), (0, -1)],      # l=5
        [(1, 1), (3, -1)]       # l=6
    ]
    
    Y_defs = [
        [(0, 1), (3, 1)],       # l=0
        [(0, 1)],               # l=1
        [(1, 1), (3, -1)],      # l=2
        [(2, 1), (0, -1)],      # l=3
        [(3, 1)],               # l=4
        [(0, 1), (1, 1)],       # l=5
        [(2, 1), (3, 1)]        # l=6
    ]

    # Inner Strassen terms definitions
    # s in 0..6
    # P_s (from X) and Q_s (from Y)
    # P_s is a linear form on X[2][2]
    # Q_s is a linear form on Y[2][2]
    
    # s=0: P = X00 + X11, Q = Y00 + Y11
    # s=1: P = X10 + X11, Q = Y00
    # s=2: P = X00,       Q = Y01 - Y11
    # s=3: P = X11,       Q = Y10 - Y00
    # s=4: P = X00 + X01, Q = Y11
    # s=5: P = X10 - X00, Q = Y00 + Y01
    # s=6: P = X01 - X11, Q = Y10 + Y11

    P_defs = [
        [(0,0, 1), (1,1, 1)],       # s=0
        [(1,0, 1), (1,1, 1)],       # s=1
        [(0,0, 1)],                 # s=2
        [(1,1, 1)],                 # s=3
        [(0,0, 1), (0,1, 1)],       # s=4
        [(1,0, 1), (0,0, -1)],      # s=5
        [(0,1, 1), (1,1, -1)]       # s=6
    ]
    
    Q_defs = [
        [(0,0, 1), (1,1, 1)],       # s=0
        [(0,0, 1)],                 # s=1
        [(0,1, 1), (1,1, -1)],      # s=2
        [(1,0, 1), (0,0, -1)],      # s=3
        [(1,1, 1)],                 # s=4
        [(0,0, 1), (0,1, 1)],       # s=5
        [(1,0, 1), (1,1, 1)]        # s=6
    ]

    # Coefficients for W
    # alpha[ij][l]
    # ij: 00, 01, 10, 11 -> index 0,1,2,3
    alpha = [[0]*7 for _ in range(4)]
    # C00 = M1 + M4 - M5 + M7 -> l=0,3,4,6
    alpha[0][0] = 1; alpha[0][3] = 1; alpha[0][4] = -1; alpha[0][6] = 1
    # C01 = M3 + M5 -> l=2,4
    alpha[1][2] = 1; alpha[1][4] = 1
    # C10 = M2 + M4 -> l=1,3
    alpha[2][1] = 1; alpha[2][3] = 1
    # C11 = M1 - M2 + M3 + M6 -> l=0,1,2,5
    alpha[3][0] = 1; alpha[3][1] = -1; alpha[3][2] = 1; alpha[3][5] = 1

    # beta[uv][s]
    # uv: 00, 01, 10, 11 -> index 0,1,2,3
    beta = [[0]*7 for _ in range(4)]
    # Z00 = N1 + N4 - N5 + N7 -> s=0,3,4,6
    beta[0][0] = 1; beta[0][3] = 1; beta[0][4] = -1; beta[0][6] = 1
    # Z01 = N3 + N5 -> s=2,4
    beta[1][2] = 1; beta[1][4] = 1
    # Z10 = N2 + N4 -> s=1,3
    beta[2][1] = 1; beta[2][3] = 1
    # Z11 = N1 - N2 + N3 + N6 -> s=0,1,2,5
    beta[3][0] = 1; beta[3][1] = -1; beta[3][2] = 1; beta[3][5] = 1

    for l in range(7):
        for s in range(7):
            c = l * 7 + s
            
            # Compute U contributions
            # P_s is linear form on X
            # X is linear form on A blocks
            # We need to accumulate coefficients for each A entry
            
            # Initialize temp arrays for A and B coefficients
            # A has 16 entries. Index 0..15
            a_coeffs = [0.0] * 16
            b_coeffs = [0.0] * 16
            
            # Process P_s
            for (u, v, coeff_p) in P_defs[s]:
                # X[u][v] is sum of A blocks
                for (block_idx, coeff_x) in X_defs[l]:
                    # Block block_idx has 4 entries
                    # Block 0: A00 (rows 0,1; cols 0,1)
                    # Block 1: A01 (rows 0,1; cols 2,3)
                    # Block 2: A10 (rows 2,3; cols 0,1)
                    # Block 3: A11 (rows 2,3; cols 2,3)
                    
                    base_row = (block_idx // 2) * 2
                    base_col = (block_idx % 2) * 2
                    
                    for du in range(2):
                        for dv in range(2):
                            # Entry in block is at (du, dv)
                            # Global row = base_row + du
                            # Global col = base_col + dv
                            # Note: The block itself is 2x2.
                            # The "entry" (u,v) in X refers to the position within the 2x2 block structure?
                            # No, X is a 2x2 matrix of blocks? No.
                            # X is a 2x2 matrix.
                            # X[u][v] is a scalar linear form?
                            # Wait. X = A00 + A11. A00 is 2x2.
                            # So X is 2x2.
                            # X[u][v] is the element at row u, col v of the matrix X.
                            # But A00 is a 2x2 matrix.
                            # So X[u][v] = (A00)[u][v] + (A11)[u][v].
                            # So for a fixed u,v, we look at A00[u][v] and A11[u][v].
                            
                            # So for P_s, we sum over u,v.
                            # For each (u,v, coeff_p), we add coeff_p * coeff_x * A_block[u][v].
                            
                            # Global index of A_block[u][v]
                            # Block block_idx is at block-row br, block-col bc.
                            br = block_idx // 2
                            bc = block_idx % 2
                            # The block occupies rows 2*br, 2*br+1 and cols 2*bc, 2*bc+1.
                            # The element (u,v) within the block is at global row 2*br+u, col 2*bc+v.
                            
                            g_row = 2 * br + u
                            g_col = 2 * bc + v
                            idx = g_row * 4 + g_col
                            
                            a_coeffs[idx] += coeff_p * coeff_x
            
            # Process Q_s
            for (u, v, coeff_q) in Q_defs[s]:
                for (block_idx, coeff_y) in Y_defs[l]:
                    br = block_idx // 2
                    bc = block_idx % 2
                    g_row = 2 * br + u
                    g_col = 2 * bc + v
                    idx = g_row * 4 + g_col
                    b_coeffs[idx] += coeff_q * coeff_y

            # Fill U and V
            for i in range(16):
                U[i][c] = a_coeffs[i]
                V[i][c] = b_coeffs[i]
            
            # Fill W
            # We need to add contributions to W for all output entries.
            # The term N_{l,s} contributes to M_{l, uv} with coeff beta[uv][s].
            # M_{l, uv} contributes to C_{ij, uv} with coeff alpha[ij][l].
            # C_{ij, uv} is at global row R = 2*ij_row + u, col C = 2*ij_col + v.
            # W index is C * 4 + R.
            
            for ij in range(4):
                ij_row = ij // 2
                ij_col = ij % 2
                coeff_a = alpha[ij][l]
                if coeff_a == 0: continue
                
                for uv in range(4):
                    u = uv // 2
                    v = uv % 2
                    coeff_b = beta[uv][s]
                    if coeff_b == 0: continue
                    
                    total_coeff = coeff_a * coeff_b
                    if total_coeff == 0: continue
                    
                    # Global R, C for C_{ij, uv}
                    R = 2 * ij_row + u
                    C = 2 * ij_col + v
                    
                    # W index
                    w_idx = C * 4 + R
                    W[w_idx][c] += total_coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
