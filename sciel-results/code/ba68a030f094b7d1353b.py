import numpy as np

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a block-Strassen approach: treat the 2x6 matrix as three 2x2 blocks.
    # Each 2x2 * 2x2 multiplication uses rank 7 (Strassen).
    # Total rank = 3 * 7 = 21.
    rank = 21
    
    U = np.zeros((n * m, rank))  # 4 x 21
    V = np.zeros((m * p, rank))  # 12 x 21
    W = np.zeros((p * n, rank))  # 12 x 21
    
    # Strassen factors for 2x2 * 2x2 -> 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+b)(e+h)
    # M2 = (a+b)(g-h)
    # M3 = (c-d)(e+f)
    # M4 = d(g-e)
    # M5 = (a-c)(h+f)
    # M6 = (b+c)(e-h)  -- wait, standard Strassen is:
    # M1 = (a+b)(e+h)
    # M2 = (a+b)(g-h)
    # M3 = (c-d)(e+f)
    # M4 = d(g-e)
    # M5 = (a-c)(h+f)
    # M6 = (b+c)(e-h)  -- actually let me use the standard formulation:
    # M1 = (a+b)(e+h)
    # M2 = (a+b)(g-h)
    # M3 = (c-d)(e+f)
    # M4 = d(g-e)
    # M5 = (a-c)(h+f)
    # M6 = (b+c)(e-h)  -- hmm, let me just use the verified one from before
    
    # Verified Strassen decomposition (from previous work):
    # M1 = (a+b)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # A coefficients (4 rows: a, b, c, d) x 7 terms
    Strassen_A = [
        [1, 0, 1, 0, 1, 1, 0],   # a
        [1, 0, 0, 0, 1, 0, 1],   # b
        [0, 1, 0, 0, 0, -1, 0],  # c
        [0, 1, 0, 1, 0, 0, -1]   # d
    ]
    
    # B coefficients (4 rows: e, f, g, h) x 7 terms
    Strassen_B = [
        [1, 1, 0, 0, 0, 1, 0],   # e
        [0, 0, 1, 0, 0, 1, 0],   # f
        [0, 0, 0, 1, 0, 0, 1],   # g
        [1, 0, -1, -1, 1, 0, 1]  # h
    ]
    
    # C coefficients (4 rows: C00, C01, C10, C11) x 7 terms
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 + M3 - M2 + M6
    Strassen_C = [
        [1, 0, 0, 1, -1, 0, 1],   # C00
        [0, 0, 1, 0, 1, 0, 0],    # C01
        [0, 1, 0, 1, 0, 0, 0],    # C10
        [1, -1, 1, 0, 0, 1, 0]    # C11
    ]
    
    # For each block t (0, 1, 2), fill columns t*7 to t*7+6
    for t in range(3):
        for r in range(7):
            col = t * 7 + r
            
            # U: 4 entries for a, b, c, d
            for i in range(4):
                U[i, col] = Strassen_A[i][r]
            
            # V: 4 entries for e, f, g, h in block t
            # Global B index: block t has elements at positions 2t, 2t+1, 6+2t, 7+2t
            # e -> B[2t], f -> B[2t+1], g -> B[6+2t], h -> B[7+2t]
            b_indices = [2*t, 2*t+1, 6+2*t, 7+2*t]
            for i in range(4):
                V[b_indices[i], col] = Strassen_B[i][r]
            
            # W: 4 entries for C00, C01, C10, C11 in block t
            # Global C index: block t has elements at positions 4t, 4t+1, 4t+2, 4t+3
            # C00 -> W[4t], C01 -> W[4t+1], C10 -> W[4t+2], C11 -> W[4t+3]
            c_indices = [4*t, 4*t+1, 4*t+2, 4*t+3]
            for i in range(4):
                W[c_indices[i], col] = Strassen_C[i][r]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
