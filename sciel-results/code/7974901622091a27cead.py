"""Matrix multiplication 2x2 x 2x8 using Strassen's rank-7 algorithm per 2x2 block."""
DIMENSIONS = [2, 2, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 8
    # p=8 means 4 independent 2x2 blocks (since p = 4*2)
    # For each 2x2 block, use Strassen's rank-7 algorithm
    # Total rank = 4 * 7 = 28
    
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]   # shape (4, 28)
    V = [[0.0] * rank for _ in range(m * p)]   # shape (16, 28)
    W = [[0.0] * rank for _ in range(p * n)]   # shape (16, 28)
    
    # Strassen's 7 products for 2x2 matrix multiplication:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B10)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    #
    # Output:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Indexing: U[i*m+j], V[j*p+k], W[k*n+i]
    # For block b (b=0..3), the k indices are 2b, 2b+1
    # i, j in {0,1}
    # U index: i*m+j = i*2+j (same for all blocks)
    # V index: j*p+k = j*8+k where k=2b or 2b+1
    # W index: k*n+i where k=2b or 2b+1, so W index = (2b)*2+i = 4b+i or (2b+1)*2+i = 4b+2+i
    
    # Strassen factors (u_i, v_j, w_{k,i}) for each of 7 products
    # u: coefficients for A[i][j], v: coefficients for B[j][k], w: coefficients for C[k][i]
    strassen = [
        # M1: (A00+A11)*(B00+B11)
        # u: A00=1, A11=1 -> U[0]=1, U[3]=1
        # v: B00=1, B11=1 -> V[0]=1, V[3]=1 (for k=0,1; j=0,1)
        # Actually let me be more careful with V indexing
        # V[j*p+k]: j is row of B, k is column of B
        # For a 2x2 block at position b: rows j=0,1; columns k=2b, 2b+1
        # B00 in block b: V[0*8+2b] = V[2b]
        # B01 in block b: V[0*8+2b+1] = V[2b+1]
        # B10 in block b: V[1*8+2b] = V[8+2b]
        # B11 in block b: V[1*8+2b+1] = V[8+2b+1]
        
        # M1: (A00+A11)*(B00+B11)
        # u: {0:1, 3:1}, v: {2b:1, 9+2b:1}, w: for C00: +1, C01: 0, C10: 0, C11: +1
        # C indices: C00 = W[2b*2+0] = W[4b], C01 = W[4b+1], C10 = W[4b+2], C11 = W[4b+3]
        # M1 contributes to C00 and C11
        ({0: 1, 3: 1}, {0: 1, 1: 1}, {0: 1, 3: 1}),  # will adjust v per block
        
        # M2: (A10+A11)*B00
        # u: {2:1, 3:1}, v: {2b:1}, w: C10: +1, C11: -1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        
        # M3: A00*(B10-B11)
        # u: {0:1}, v: {8+2b:1, 9+2b:-1}, w: C01: +1, C11: +1
        ({0: 1}, {0: 1, 1: -1}, {1: 1, 3: 1}),
        
        # M4: A11*(B01-B10)
        # u: {3:1}, v: {2b+1:1, 8+2b:-1}, w: C00: +1, C10: +1
        ({3: 1}, {0: -1, 1: 1}, {0: 1, 2: 1}),
        
        # M5: (A00+A01)*B11
        # u: {0:1, 1:1}, v: {9+2b:1}, w: C00: -1, C01: +1
        ({0: 1, 1: 1}, {1: 1}, {0: -1, 1: 1}),
        
        # M6: (A10-A00)*(B00+B10)
        # u: {0:-1, 2:1}, v: {2b:1, 8+2b:1}, w: C11: +1
        ({0: -1, 2: 1}, {0: 1, 1: 1}, {3: 1}),
        
        # M7: (A01-A11)*(B01+B11)
        # u: {1:1, 3:-1}, v: {2b+1:1, 9+2b:1}, w: C00: +1
        ({1: 1, 3: -1}, {0: 1, 1: 1}, {0: 1}),
    ]
    
    # Now build U, V, W
    for b in range(4):
        col_base = b * 7
        for s in range(7):
            col = col_base + s
            u_dict, v_dict, w_dict = strassen[s]
            # U: same for all blocks
            for ui, val in u_dict.items():
                U[ui][col] = float(val)
            # V: j*p + (2b + k_offset) where k_offset is 0 or 1
            for vj, val in v_dict.items():
                # v_dict keys are k_offset (0 or 1), value is coefficient
                # But wait, v_dict has keys that represent B column offset within the 2x2 block
                # For M1: v = {0: 1, 1: 1} means B00 and B11? No...
                # Let me re-examine. In my strassen list, v_dict keys should map to B positions
                # Actually I need to restructure. Let me redo this more carefully.
                pass
    
    # Let me rebuild this more carefully
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # For each Strassen term s (0..6) and each block b (0..3):
    # col = b*7 + s
    # U coefficients: same for all b, from u_coeffs[s]
    # V coefficients: for B[j][2b+k], from v_coeffs[s] which maps (j,k) -> coeff
    # W coefficients: for C[2b+k][i], from w_coeffs[s] which maps (k,i) -> coeff
    
    # Define Strassen terms with explicit index mappings
    # u_coeffs[s]: dict from A index (i*m+j) to coeff
    # v_coeffs[s]: dict from (j, k_offset) to coeff, where k_offset in {0,1}
    # w_coeffs[s]: dict from (k_offset, i) to coeff, where k_offset in {0,1}
    
    u_coeffs = [
        {0: 1, 3: 1},       # M1: A00 + A11
        {2: 1, 3: 1},       # M2: A10 + A11
        {0: 1},             # M3: A00
        {3: 1},             # M4: A11
        {0: 1, 1: 1},       # M5: A00 + A01
        {0: -1, 2: 1},      # M6: A10 - A00
        {1: 1, 3: -1},      # M7: A01 - A11
    ]
    
    v_coeffs = [
        {(0,0): 1, (1,1): 1},      # M1: B00 + B11
        {(0,0): 1},                 # M2: B00
        {(1,0): 1, (1,1): -1},     # M3: B10 - B11
        {(0,1): 1, (1,0): -1},     # M4: B01 - B10
        {(1,1): 1},                 # M5: B11
        {(0,0): 1, (1,0): 1},      # M6: B00 + B10
        {(0,1): 1, (1,1): 1},      # M7: B01 + B11
    ]
    
    w_coeffs = [
        {(0,0): 1, (1,1): 1},      # M1: C00 + C11
        {(1,0): 1, (1,1): -1},     # M2: C10 - C11
        {(1,1): 1, (1,0): 1},      # M3: C01 + C11  (wait, C[k][i], k is row of C)
        {(0,0): 1, (1,0): 1},      # M4: C00 + C10
        {(0,0): -1, (0,1): 1},     # M5: -C00 + C01
        {(1,1): 1},                 # M6: C11
        {(0,0): 1},                 # M7: C00
    ]
    
    # Wait, I need to double-check the Strassen formulas with the correct indexing
    # Standard Strassen:
    # C = [[C00, C01], [C10, C11]]
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # In our convention, C[k*n+i] where k is the "output column index" (from B's columns) and i is the output row
    # So C00 = C[0][0] -> W[0*2+0] = W[0]
    # C01 = C[0][1] -> W[0*2+1] = W[1]
    # C10 = C[1][0] -> W[1*2+0] = W[2]
    # C11 = C[1][1] -> W[1*2+1] = W[3]
    #
    # For block b, C[k][i] -> W[(2b+k)*2 + i]
    
    w_coeffs = [
        {(0,0): 1, (1,1): 1},      # M1: C[0][0] + C[1][1]
        {(1,0): 1, (1,1): -1},     # M2: C[1][0] - C[1][1]
        {(0,1): 1, (1,1): 1},      # M3: C[0][1] + C[1][1]
        {(0,0): 1, (1,0): 1},      # M4: C[0][0] + C[1][0]
        {(0,0): -1, (0,1): 1},     # M5: -C[0][0] + C[0][1]
        {(1,1): 1},                 # M6: C[1][1]
        {(0,0): 1},                 # M7: C[0][0]
    ]
    
    for b in range(4):
        for s in range(7):
            col = b * 7 + s
            # Fill U
            for a_idx, coeff in u_coeffs[s].items():
                U[a_idx][col] = float(coeff)
            # Fill V
            for (j, k_off), coeff in v_coeffs[s].items():
                v_idx = j * p + (2 * b + k_off)
                V[v_idx][col] = float(coeff)
            # Fill W
            for (k_off, i), coeff in w_coeffs[s].items():
                w_idx = (2 * b + k_off) * n + i
                W[w_idx][col] = float(coeff)
    
    return {"U": U, "V": V, "W": W}
