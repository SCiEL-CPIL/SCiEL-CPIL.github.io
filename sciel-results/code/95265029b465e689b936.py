import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = DIMENSIONS
    
    # We use a block-wise Strassen decomposition.
    # The problem is 2x4 * 4x6.
    # We split the 6 columns into three 2-column blocks.
    # Each block is a 2x4 * 4x2 multiplication.
    # We use Strassen's algorithm for 2x2 * 2x2 twice per block?
    # No, 2x4 * 4x2 can be done in rank 14 (two 2x2 Strassen).
    # Total rank = 3 * 14 = 42.
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    # Strassen coefficients for 2x2 * 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a-b)(f+h)
    # M4 = (a+d)(h-g)
    # M5 = (a+c)(e-f)
    # M6 = (b+d)(g-e)
    # M7 = (a-c)(f+g)
    
    U_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1, (0, 1): -1},
        {(0, 0): 1, (1, 1): 1, (1, 0): -1},
        {(0, 0): 1, (1, 0): 1, (0, 1): -1},
        {(0, 1): 1, (1, 1): 1, (1, 0): -1},
        {(0, 0): 1, (1, 0): -1, (0, 1): 1}
    ]
    
    V_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(0, 1): 1, (1, 1): 1},
        {(1, 1): 1, (1, 0): -1},
        {(0, 0): 1, (0, 1): -1},
        {(1, 0): 1, (0, 0): -1},
        {(0, 1): 1, (1, 0): 1}
    ]
    
    W_contrib = [
        {(0, 0): 1, (1, 1): 1, (0, 1): -1, (1, 0): -1},
        {(0, 0): -1, (0, 1): 1, (1, 0): 1, (1, 1): -1},
        {(0, 0): 1, (1, 1): 1, (0, 1): -1, (1, 0): -1},
        {(0, 0): 1, (1, 1): 1, (0, 1): -1, (1, 0): -1},
        {(0, 0): 1, (1, 1): 1, (0, 1): -1, (1, 0): -1},
        {(0, 0): 1, (1, 1): 1, (0, 1): -1, (1, 0): -1},
        {(0, 0): 1, (1, 1): 1, (0, 1): -1, (1, 0): -1}
    ]
    # Correct W for Strassen:
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M6 + M7
    
    W_contrib = [
        {(0, 0): 1, (1, 1): 1, (0, 1): -1, (1, 0): -1}, # M1
        {(0, 0): -1, (0, 1): 1, (1, 0): 1, (1, 1): -1}, # M2 -> C21 has M2, C11 has -M2, C22 has -M2? No.
        # Let's re-verify Strassen W.
        # C = [[M1+M4-M5-M7, M3+M5], [M2+M4, M1-M2+M6+M7]]
        # M1: C00:1, C11:1
        # M2: C10:1, C11:-1
        # M3: C01:1
        # M4: C00:1, C10:1
        # M5: C00:-1, C01:1
        # M6: C11:1
        # M7: C00:-1, C11:1
        
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): -1},
        {(0, 1): 1},
        {(0, 0): 1, (1, 0): 1},
        {(0, 0): -1, (0, 1): 1},
        {(1, 1): 1},
        {(0, 0): -1, (1, 1): 1}
    ]

    col = 0
    for a in range(2): # 2 row blocks of A? No, A is 2x4. We split A into two 2x2 blocks horizontally.
        for b in range(3): # 3 column blocks of B (each 2 cols)
            for s in range(7):
                if time.time() > deadline:
                    break
                
                # U: A is 2x4. Block a is columns 2a, 2a+1.
                # A_local is 2x2.
                # U_coeffs[s] has keys (i_loc, j_loc).
                # i_loc is row 0 or 1.
                # j_loc is col 0 or 1 within the block.
                for (i_loc, j_loc), coeff in U_coeffs[s].items():
                    i = i_loc
                    j = a * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                
                # V: B is 4x6. Block b is rows 2a, 2a+1? No.
                # B is 4x6. We are multiplying A (2x4) by B_block (4x2).
                # B_block rows correspond to A's columns.
                # A's columns are 0,1,2,3.
                # Block a of A uses columns 2a, 2a+1.
                # So B_block must use rows 2a, 2a+1?
                # No, B is 4x6. The rows of B correspond to the columns of A.
                # So if A uses columns 0,1, B must use rows 0,1.
                # If A uses columns 2,3, B must use rows 2,3.
                # So for block a, we use rows 2a, 2a+1 of B.
                # And columns 2b, 2b+1 of B (since p=6, 3 blocks of 2).
                
                for (j_loc, k_loc), coeff in V_coeffs[s].items():
                    j = a * 2 + j_loc
                    k = b * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                
                # W: C is 2x6.
                # Output block is rows 0,1 and cols 2b, 2b+1.
                for (i_loc, k_loc), coeff in W_contrib[s].items():
                    i = i_loc
                    k = b * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
