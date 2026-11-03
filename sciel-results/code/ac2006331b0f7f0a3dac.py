"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # U: (n*m, rank) = (8, 56)
    # V: (m*p, rank) = (32, 56)
    # W: (p*n, rank) = (16, 56)
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2 multiplication
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # Output reconstruction:
    # M11 = M1 + M4 - M5 + M7
    # M12 = M3 + M5
    # M21 = M2 + M4
    # M22 = M1 - M2 + M3 + M6
    
    # Define the 7 Strassen steps as tuples of (A_coeffs, B_coeffs, W_coeffs)
    # A_coeffs: dict mapping A_local_idx (0..3) to coeff
    # B_coeffs: dict mapping B_local_idx (0..3) to coeff
    # W_coeffs: dict mapping C_local_idx (0..3) to coeff
    
    # Local indices for 2x2 matrix:
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    strassen_steps = [
        # M1: (a+d)(e+h)
        {
            'A': {0: 1, 3: 1},
            'B': {0: 1, 3: 1},
            'W': {0: 1, 1: 0, 2: 0, 3: 1} # M11, M22
        },
        # M2: (c+d)e
        {
            'A': {2: 1, 3: 1},
            'B': {0: 1},
            'W': {2: 1, 3: -1} # M21, M22
        },
        # M3: a(f-h)
        {
            'A': {0: 1},
            'B': {1: 1, 3: -1},
            'W': {1: 1, 3: 1} # M12, M22
        },
        # M4: d(g-e)
        {
            'A': {3: 1},
            'B': {2: 1, 0: -1},
            'W': {0: 1, 2: 1} # M11, M21
        },
        # M5: (a+b)h
        {
            'A': {0: 1, 1: 1},
            'B': {3: 1},
            'W': {0: -1, 1: 1} # M11, M12
        },
        # M6: (c-a)(e+f)
        {
            'A': {2: 1, 0: -1},
            'B': {0: 1, 1: 1},
            'W': {3: 1} # M22
        },
        # M7: (b-d)(g+h)
        {
            'A': {1: 1, 3: -1},
            'B': {2: 1, 3: 1},
            'W': {0: 1} # M11
        }
    ]
    
    # Iterate over the 4 column blocks of the output C
    for c_block in range(4):
        # Process A1 * B_top_block
        # A1 is A[0:2, 0:2]
        # B_top_block is B[0:2, 4*c_block : 4*c_block + 2]
        
        for step in strassen_steps:
            # Fill U
            for a_idx, coeff in step['A'].items():
                # a_idx 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
                r, c_loc = divmod(a_idx, 2)
                # A1 global indices: row r, col c_loc (since A1 is at cols 0,1)
                u_row = r * m + c_loc
                U[u_row][col] += coeff
                
            # Fill V
            for b_idx, coeff in step['B'].items():
                r, c_loc = divmod(b_idx, 2)
                # B_top_block global indices: row r (0 or 1), col 4*c_block + c_loc
                v_row = r * p + (4 * c_block + c_loc)
                V[v_row][col] += coeff
                
            # Fill W
            for w_idx, coeff in step['W'].items():
                r, c_loc = divmod(w_idx, 2)
                # C block is at cols 4*c_block + c_loc
                # W index convention: k*n + i, where k is row of C, i is col of C
                # Here C block is 2x2.
                # r is row in block (0,1), c_loc is col in block (0,1)
                # Global row k = r (since C rows are 0..1 for this block? No, C is 2x8.
                # Wait, C is 2x8.
                # The block C[0:2, 4*c_block:4*c_block+2] occupies rows 0,1 of C.
                # So k = r. i = 4*c_block + c_loc.
                w_row = r * n + (4 * c_block + c_loc)
                W[w_row][col] += coeff
                
            col += 1
            
        # Process A2 * B_bot_block
        # A2 is A[0:2, 2:4]
        # B_bot_block is B[2:4, 4*c_block : 4*c_block + 2]
        
        for step in strassen_steps:
            # Fill U
            for a_idx, coeff in step['A'].items():
                r, c_loc = divmod(a_idx, 2)
                # A2 global indices: row r, col 2 + c_loc
                u_row = r * m + (2 + c_loc)
                U[u_row][col] += coeff
                
            # Fill V
            for b_idx, coeff in step['B'].items():
                r, c_loc = divmod(b_idx, 2)
                # B_bot_block global indices: row 2 + r, col 4*c_block + c_loc
                v_row = (2 + r) * p + (4 * c_block + c_loc)
                V[v_row][col] += coeff
                
            # Fill W
            for w_idx, coeff in step['W'].items():
                r, c_loc = divmod(w_idx, 2)
                # C block is at cols 4*c_block + c_loc
                # Global row k = r. i = 4*c_block + c_loc.
                w_row = r * n + (4 * c_block + c_loc)
                W[w_row][col] += coeff
                
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
