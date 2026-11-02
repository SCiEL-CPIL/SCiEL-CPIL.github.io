import time

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2x2
    # We have 8 blocks of 2 columns in B, corresponding to 2 rows in C.
    # Block b corresponds to B columns 2b, 2b+1 and C rows 2b, 2b+1.
    
    # Indices in U: 0: A00, 1: A01, 2: A10, 3: A11
    # Indices in V for block b:
    #   B00: 0*16 + 2b = 2b
    #   B01: 0*16 + 2b+1 = 2b+1
    #   B10: 1*16 + 2b = 16+2b
    #   B11: 1*16 + 2b+1 = 16+2b+1
    # Indices in W for block b:
    #   C00: 2b*2 + 0 = 4b
    #   C01: (2b+1)*2 + 0 = 4b+2
    #   C10: 2b*2 + 1 = 4b+1
    #   C11: (2b+1)*2 + 1 = 4b+3
    
    # Define Strassen terms (s, u_coeff, v_coeff, w_coeff)
    # u_coeff: list of (U_row, coeff)
    # v_coeff: list of (V_row_offset, coeff) -> actual row = offset + base
    # w_coeff: list of (W_row_offset, coeff) -> actual row = offset + base
    
    # s=0: M1 = (A00+A11)(B00+B11)
    term0 = {
        'u': [(0, 1), (3, 1)],
        'v': [(0, 1), (1, 1)], # Offsets relative to 2b and 16+2b? No, let's use explicit offsets from 2b
        'w': [(0, 1), (3, 1)]  # Offsets relative to 4b
    }
    # s=1: M2 = (A10+A11)B00
    term1 = {
        'u': [(2, 1), (3, 1)],
        'v': [(0, 1)],
        'w': [(1, 1), (3, -1)]
    }
    # s=2: M3 = A00(B01-B11)
    term2 = {
        'u': [(0, 1)],
        'v': [(1, 1), (1, -1)], # B01 offset 1, B11 offset 1 (but row 16+2b)
        'w': [(2, 1), (3, 1)]
    }
    # s=3: M4 = A11(B10-B00)
    term3 = {
        'u': [(3, 1)],
        'v': [(1, 1), (0, -1)], # B10 offset 1 (row 16+2b), B00 offset 0 (row 2b)
        'w': [(0, 1), (1, 1)]
    }
    # s=4: M5 = (A00+A01)B11
    term4 = {
        'u': [(0, 1), (1, 1)],
        'v': [(1, 1)], # B11
        'w': [(0, -1), (2, 1)]
    }
    # s=5: M6 = (A10-A00)(B00+B01)
    term5 = {
        'u': [(2, 1), (0, -1)],
        'v': [(0, 1), (1, 1)],
        'w': [(3, 1)]
    }
    # s=6: M7 = (A01-A11)(B10+B11)
    term6 = {
        'u': [(1, 1), (3, -1)],
        'v': [(1, 1), (1, 1)],
        'w': [(0, 1)]
    }
    
    # We need to be careful with V indices.
    # B00: row 2b
    # B01: row 2b+1
    # B10: row 16+2b
    # B11: row 16+2b+1
    
    # Let's refine the V index mapping in the loop.
    
    for b in range(8):
        base_v_b0 = 2 * b
        base_v_b1 = 16 + 2 * b
        base_w = 4 * b
        
        for s in range(7):
            r = b * 7 + s
            if s == 0:
                # M1
                U[0][r] += 1; U[3][r] += 1
                V[base_v_b0][r] += 1; V[base_v_b1 + 1][r] += 1 # B00, B11
                W[base_w][r] += 1; W[base_w + 3][r] += 1 # C00, C11
            elif s == 1:
                # M2
                U[2][r] += 1; U[3][r] += 1
                V[base_v_b0][r] += 1
                W[base_w + 1][r] += 1; W[base_w + 3][r] += -1 # C10, C11
            elif s == 2:
                # M3
                U[0][r] += 1
                V[base_v_b0 + 1][r] += 1; V[base_v_b1 + 1][r] += -1 # B01, B11
                W[base_w + 2][r] += 1; W[base_w + 3][r] += 1 # C01, C11
            elif s == 3:
                # M4
                U[3][r] += 1
                V[base_v_b1][r] += 1; V[base_v_b0][r] += -1 # B10, B00
                W[base_w][r] += 1; W[base_w + 1][r] += 1 # C00, C10
            elif s == 4:
                # M5
                U[0][r] += 1; U[1][r] += 1
                V[base_v_b1 + 1][r] += 1 # B11
                W[base_w][r] += -1; W[base_w + 2][r] += 1 # C00, C01
            elif s == 5:
                # M6
                U[2][r] += 1; U[0][r] += -1
                V[base_v_b0][r] += 1; V[base_v_b0 + 1][r] += 1 # B00, B01
                W[base_w + 3][r] += 1 # C11
            elif s == 6:
                # M7
                U[1][r] += 1; U[3][r] += -1
                V[base_v_b1][r] += 1; V[base_v_b1 + 1][r] += 1 # B10, B11
                W[base_w][r] += 1 # C00
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
