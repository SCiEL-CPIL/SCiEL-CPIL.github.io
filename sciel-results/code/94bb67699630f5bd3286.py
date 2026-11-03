import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1: (a+d)(e+h) -> C11+=1, C22+=1
    # M2: (c+d)e -> C21+=1
    # M3: a(f-h) -> C12+=1, C22+=1
    # M4: d(g-e) -> C11+=1, C21+=1
    # M5: (a+b)h -> C11-=1, C12+=1
    # M6: (c-a)(e+f) -> C22+=1
    # M7: (b-d)(g+h) -> C11+=1
    
    # Define the 7 Strassen terms
    # Each term is a list of (U_indices, coeff), (V_indices, coeff), (W_indices, coeff)
    # U indices are (i, j_local)
    # V indices are (j_local, k_local)
    # W indices are (i, k_local)
    
    strassen_terms = [
        # M1
        {
            'U': { (0,0): 1, (1,1): 1 },
            'V': { (0,0): 1, (1,1): 1 },
            'W': { (0,0): 1, (1,1): 1 }
        },
        # M2
        {
            'U': { (1,0): 1, (1,1): 1 },
            'V': { (0,0): 1 },
            'W': { (1,0): 1 }
        },
        # M3
        {
            'U': { (0,0): 1 },
            'V': { (0,1): 1, (1,1): -1 },
            'W': { (0,1): 1, (1,1): 1 }
        },
        # M4
        {
            'U': { (1,1): 1 },
            'V': { (1,0): 1, (0,0): -1 },
            'W': { (0,0): 1, (1,0): 1 }
        },
        # M5
        {
            'U': { (0,0): 1, (0,1): 1 },
            'V': { (1,1): 1 },
            'W': { (0,0): -1, (0,1): 1 }
        },
        # M6
        {
            'U': { (1,0): 1, (0,0): -1 },
            'V': { (0,0): 1, (0,1): 1 },
            'W': { (1,1): 1 }
        },
        # M7
        {
            'U': { (0,1): 1, (1,1): -1 },
            'V': { (1,0): 1, (1,1): 1 },
            'W': { (0,0): 1 }
        }
    ]
    
    # Define the 6 blocks
    # (a_cols_offset, b_rows_offset, b_cols_offset, c_cols_offset)
    blocks = [
        (0, 0, 0, 0), # A_L B_L1
        (0, 0, 2, 2), # A_L B_L2
        (0, 0, 4, 4), # A_L B_L3
        (2, 2, 0, 0), # A_R B_R1
        (2, 2, 2, 2), # A_R B_R2
        (2, 2, 4, 4), # A_R B_R3
    ]
    
    col = 0
    for a_off, b_row_off, b_col_off, c_col_off in blocks:
        for term in strassen_terms:
            # Fill U
            for (i, j_loc), coeff in term['U'].items():
                global_i = i
                global_j = a_off + j_loc
                U[global_i * m + global_j][col] += coeff
            
            # Fill V
            for (j_loc, k_loc), coeff in term['V'].items():
                global_j = b_row_off + j_loc
                global_k = b_col_off + k_loc
                V[global_j * p + global_k][col] += coeff
            
            # Fill W
            for (i, k_loc), coeff in term['W'].items():
                global_i = i
                global_k = c_col_off + k_loc
                W[global_k * n + global_i][col] += coeff
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
