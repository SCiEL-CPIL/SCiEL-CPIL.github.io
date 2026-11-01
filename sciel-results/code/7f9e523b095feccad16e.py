import time

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=12
    # U shape (n*m, rank) = (4, rank)
    # V shape (m*p, rank) = (24, rank)
    # W shape (p*n, rank) = (24, rank)
    
    # We use 6 blocks of 2 columns each.
    # Each block uses Strassen's algorithm for 2x2 by 2x2.
    # Rank per block = 7.
    # Total rank = 6 * 7 = 42.
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2 by 2x2
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # U coefficients for a, b, c, d
    # V coefficients for e, f, g, h (local to block)
    # W coefficients for C00, C01, C10, C11 (local to block)
    
    # Local indices for A: 0:a, 1:b, 2:c, 3:d
    # Local indices for B: 0:e, 1:f, 2:g, 3:h
    # Local indices for C: 0:C00, 1:C01, 2:C10, 3:C11
    
    strassen_steps = [
        # (U_coefs, V_coefs, W_coefs)
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 0]), # C00 += M1
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]), # C10 += M2
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]), # C01 += M3
        # M4
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]), # C00 += M4, C10 += M4
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # C00 -= M5, C01 += M5
        # M6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # C11 += M6
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1]), # C00 += M7, C11 += M7
    ]
    
    # Note: The W_coefs above are accumulated.
    # Let's define the reconstruction explicitly to be safe.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We will fill U, V, W column by column.
    col_idx = 0
    for block in range(6):
        # B local indices in global V:
        # e = B[0, 2*block] -> index 2*block
        # f = B[0, 2*block+1] -> index 2*block+1
        # g = B[1, 2*block] -> index 12 + 2*block
        # h = B[1, 2*block+1] -> index 12 + 2*block+1
        
        v_e = 2 * block
        v_f = 2 * block + 1
        v_g = 12 + 2 * block
        v_h = 12 + 2 * block + 1
        
        # C local indices in global W:
        # C00 = C[0, 2*block] -> index 2*block
        # C01 = C[0, 2*block+1] -> index 2*block+1
        # C10 = C[1, 2*block] -> index 12 + 2*block
        # C11 = C[1, 2*block+1] -> index 12 + 2*block+1
        
        w_c00 = 2 * block
        w_c01 = 2 * block + 1
        w_c10 = 12 + 2 * block
        w_c11 = 12 + 2 * block + 1
        
        # 7 steps for this block
        for u_coefs, v_coefs, w_coefs in strassen_steps:
            # Fill U
            for i in range(4):
                U[i][col_idx] = u_coefs[i]
            
            # Fill V
            V[v_e][col_idx] = v_coefs[0]
            V[v_f][col_idx] = v_coefs[1]
            V[v_g][col_idx] = v_coefs[2]
            V[v_h][col_idx] = v_coefs[3]
            
            # Fill W
            W[w_c00][col_idx] = w_coefs[0]
            W[w_c01][col_idx] = w_coefs[1]
            W[w_c10][col_idx] = w_coefs[2]
            W[w_c11][col_idx] = w_coefs[3]
            
            col_idx += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
